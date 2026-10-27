"""Calibrate synthetic DGF/SA default reservoir designs to site-grouped cost Q50.

Run from the repository root:
    pixi run python scripts/experiments/calibrate_reference_design.py `
        --samples outputs/cost_sensitivity/reservoir_sample_results.csv `
        --groups data/resources/reservoir_geographic_groups.csv

This script NEVER edits package defaults. Costs are reevaluated using the exact
reference-case configuration and functions defined in cost_sensitivity.py.
Requires scipy and pandas in the pixi environment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# When invoked as scripts/experiments/calibrate_reservoir_reference.py, its
# sibling directory is on sys.path. Importing this does NOT run its main().
import cost_sensitivity as cs

COSTS = ('injection_tcr_rate', 'storage_tcr_rate', 'withdrawal_tcr_rate')
PRESSURES = ('model_abandonment_pressure_bar', 'min_storage_pressure_bar', 'max_storage_pressure_bar')
CLASS_TO_TECH = {'DGF': 'depleted_gas_field', 'SA': 'aquifer'}


def weighted_quantile(values, weights, quantile=0.5):
    """Continuous midpoint-of-weight quantile (a stable weighted Q50 convention)."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[good], w[good]
    if not len(v):
        raise ValueError('No data for quantile')
    order = np.argsort(v)
    v, w = v[order], w[order]
    centres = (np.cumsum(w) - 0.5*w) / sum(w)
    return float(np.interp(quantile, centres, v))


def attach_groups(samples, group_file):
    """Import explicitly reviewed spatial groups, otherwise collapse only exact site duplicates."""
    if group_file is None:
        samples['geographic_group'] = samples['country'].astype(str) + ' / ' + samples['site'].astype(str)
        print('WARNING: no --groups supplied; only identical site names are grouped. '
              'Supply reviewed groups before adopting calibrated defaults.')
        return samples
    groups = pd.read_csv(group_file, dtype={'source_site_id': str})
    required = {'storage_class', 'site', 'geographic_group'}
    if not required.issubset(groups.columns):
        raise ValueError(f'Group file needs {sorted(required)}')
    groups = groups[list(required)].drop_duplicates()
    if groups.duplicated(['storage_class', 'site']).any():
        raise ValueError('Geographic groups ambiguous for some storage_class/site pairs')
    samples = samples.merge(groups, on=['storage_class','site'], how='left', validate='many_to_one')
    missing = samples.geographic_group.isna()
    if missing.any():
        print(f'WARNING: {missing.sum()} rows have no group assignment; treating each site separately')
        samples.loc[missing, 'geographic_group'] = (
            samples.loc[missing, 'country'].astype(str) + ' / ' + samples.loc[missing,'site'].astype(str)
        )
    return samples


def weighted_cost_targets(df):
    counts = df.groupby('geographic_group')['geographic_group'].transform('size').astype(float)
    weights = 1.0/counts  # each distinct group has total weight one
    return np.array([weighted_quantile(df[c], weights) for c in COSTS]), weights


def joint_distance(rates, targets):
    rates = np.asarray(rates, dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        return 1e6
    return float(np.sqrt(np.sum(np.log(rates / targets)**2)))


def model_rates(technology, reference, x):
    """Reconstruct only reservoir design variables; other paper assumptions fixed."""
    depth = float(x[0])
    if technology.technology == 'aquifer':
        # hydrostatic gradient * pressure multiplier, pressure in bar
        pmin = depth * 0.1 * float(x[1])
        pmax = pmin * float(x[2])
        pab = None
    else:
        pab, pmin, pmax = [float(v) for v in x[1:]]
    if not (0 < pmin < pmax) or (pab is not None and not 0 < pab < pmin):
        raise ValueError('Invalid pressure ordering')
    general = reference[0].general
    updates = dict(reference_reservoir_depth_m=depth,
                   minimum_operating_pressure_pa=pmin*1e5,
                   maximum_operating_pressure_pa=pmax*1e5,
                   temperature_k=cs.calculate_temperature_from_depth_k(depth))
    if pab is not None:
        updates['abandonment_pressure_pa'] = pab*1e5
    assumptions = replace(reference[0], general=replace(general, **updates))
    project = cs.construct_reference_project(technology, assumptions)
    breakdown = cs.calculate_cost_breakdown(
        technology, project, hystories_overrides=reference[1], lrc_overrides=reference[2])
    rates = cs.summarise_tcr_rates(breakdown)
    return np.array([rates[c.split('_')[0].capitalize()] for c in COSTS], dtype=float)


def fit_one(cls, samples, maxiter, nstarts):
    d = samples[(samples.storage_class == cls) & (samples.status == 'ok')].copy()
    for col in COSTS + ('depth_m_representative','min_storage_pressure_bar','max_storage_pressure_bar'):
        d[col] = pd.to_numeric(d[col], errors='coerce')
    if cls == 'DGF':
        d[PRESSURES[0]] = pd.to_numeric(d[PRESSURES[0]], errors='coerce')
        d = d.dropna(subset=list(PRESSURES) + ['depth_m_representative', *COSTS])
        d = d[(d[PRESSURES[0]] > 0) & (d[PRESSURES[0]] < d[PRESSURES[1]]) &
              (d[PRESSURES[1]] < d[PRESSURES[2]])]
    else:
        d = d.dropna(subset=['depth_m_representative','min_storage_pressure_bar','max_storage_pressure_bar',*COSTS])
        d = d[(d.min_storage_pressure_bar > 0) & (d.min_storage_pressure_bar < d.max_storage_pressure_bar)]
    if len(d) < 4:
        raise ValueError(f'Insufficient complete {cls} configurations: {len(d)}')
    targets, weights = weighted_cost_targets(d)
    d['joint_log_distance'] = [joint_distance(row, targets) for row in d[list(COSTS)].to_numpy()]
    medoid = d.sort_values('joint_log_distance').iloc[0]
    tech = next(t for t in cs.TECHNOLOGIES if t.technology == CLASS_TO_TECH[cls])
    reference = cs.build_case_configuration(tech)
    if cls == 'SA':
        # Tie pressure to the hydrostatic pressure derived from depth. Observed
        # regional pressure multipliers and Pmax/Pmin ratios define broad bounds.
        grad = (d.min_storage_pressure_bar / (d.depth_m_representative*0.1)).to_numpy()
        ratio = (d.max_storage_pressure_bar / d.min_storage_pressure_bar).to_numpy()
        medoid_x = np.array([medoid.depth_m_representative,
                             medoid.min_storage_pressure_bar/(0.1*medoid.depth_m_representative),
                             medoid.max_storage_pressure_bar/medoid.min_storage_pressure_bar])
        # Exclude very unusual gradients from optimisation and constrain to
        # reasonably hydrostatic conditions, without excluding site cases from Q50.
        bounds = [(float(d.depth_m_representative.quantile(.10)),float(d.depth_m_representative.quantile(.90))),
                  (max(.90,float(np.quantile(grad,.10))),min(1.10,float(np.quantile(grad,.90)))),
                  (max(1.03,float(np.quantile(ratio,.10))),float(np.quantile(ratio,.90)))]
    else:
        medoid_x = medoid[['depth_m_representative', *PRESSURES]].to_numpy(dtype=float)
        bounds = [(float(d[c].quantile(.05)),float(d[c].quantile(.95)))
                  for c in ('depth_m_representative',*PRESSURES)]
    if any(lo >= hi for lo,hi in bounds):
        raise ValueError(f'Invalid optimisation bounds for {cls}: {bounds}')
    centre = np.array([(lo+hi)/2 for lo,hi in bounds])
    scale = np.array([hi-lo for lo,hi in bounds])
    cache = {}
    def evaluate(y):
        x = np.clip(np.asarray(y), 0, 1)*scale+np.array([b[0] for b in bounds])
        key = tuple(np.round(x, 7))
        if key not in cache:
            try:
                rates = model_rates(tech,reference,x)
                cache[key] = (joint_distance(rates,targets),rates)
            except (ValueError, ArithmeticError, OverflowError) as exc:
                cache[key] = (1e5,None)
        return cache[key]
    def objective(y):
        x = np.asarray(y)*scale+np.array([b[0] for b in bounds])
        if cls == 'DGF':
            pab,pmin,pmax = x[1:]
            if pab + 1 >= pmin or pmin + 1 >= pmax:
                return 1e5 + 100*max(pab+1-pmin,0)**2 + 100*max(pmin+1-pmax,0)**2
        score,_ = evaluate(y)
        return score
    x0s = [medoid_x,centre]
    sorted_d = d.sort_values('joint_log_distance')
    for _,row in sorted_d.head(max(nstarts-2,0)).iterrows():
        if cls == 'SA':
            x0s.append(np.array([row.depth_m_representative,
                                 row.min_storage_pressure_bar/(.1*row.depth_m_representative),
                                 row.max_storage_pressure_bar/row.min_storage_pressure_bar]))
        else:
            x0s.append(row[['depth_m_representative',*PRESSURES]].to_numpy(float))
    lower = np.array([b[0] for b in bounds])
    solutions=[]
    for x0 in x0s[:nstarts]:
        y0 = np.clip((x0-lower)/scale,0,1)
        result = minimize(objective,y0,method='Powell',bounds=[(0,1)]*len(bounds),
                          options={'maxiter':maxiter,'maxfev':maxiter*18,'xtol':1e-3,'ftol':1e-4})
        x = np.clip(result.x,0,1)*scale+lower
        score, rates=evaluate(result.x)
        if rates is not None:
            solutions.append((score,x,rates,result.success,result.nfev))
    if not solutions:
        raise RuntimeError(f'No feasible optimised {cls} configuration')
    score,x,rates,success,nfev=min(solutions,key=lambda a:a[0])
    if cls == 'SA':
        pmin = x[0]*.1*x[1]; pmax=pmin*x[2]; pab=np.nan
    else:
        pab,pmin,pmax=x[1:]
    summary = {'storage_class':cls,'n_rows':len(d),'n_geographic_groups':d.geographic_group.nunique(),
               'reference_site':str(medoid.site),'reference_site_score':medoid.joint_log_distance,
               'optimised_score':score,'depth_m':x[0], 'abandonment_pressure_bar':pab,
               'minimum_pressure_bar':pmin, 'maximum_pressure_bar':pmax,
               'optimiser_success':success,'function_evaluations_last_start':nfev}
    for c,target,actual in zip(COSTS,targets,rates):
        summary[f'q50_{c}']=target
        summary[f'calibrated_{c}']=actual
        summary[f'error_pct_{c}']=100*(actual/target-1)
        summary[f'medoid_{c}']=float(medoid[c])
    d['geographic_group_weight']=weights
    return summary,d.sort_values('joint_log_distance')



# Empirical parameter quantiles are descriptive screening bounds, not
# jointly observed reservoir configurations. Each geographical group receives
# equal total weight, split among its constituent site rows.
PARAMETERS = {
    'DGF': (
        ('depth_m_representative', 'depth_m', 'm'),
        ('model_abandonment_pressure_bar', 'abandonment_pressure_bar', 'bar'),
        ('min_storage_pressure_bar', 'minimum_pressure_bar', 'bar'),
        ('max_storage_pressure_bar', 'maximum_pressure_bar', 'bar'),
    ),
    'SA': (
        ('depth_m_representative', 'depth_m', 'm'),
        ('min_storage_pressure_bar', 'minimum_pressure_bar', 'bar'),
        ('max_storage_pressure_bar', 'maximum_pressure_bar', 'bar'),
    ),
}
QUANTILES = (10, 25, 50, 75, 90)


def parameter_quantile_rows(cls, ranked, summary):
    """Geography-weighted empirical marginal quantiles, with OAT feasibility flags.

    Pressure feasibility is tested by replacing ONLY the selected field in the
    calibrated reference. No quantile is clipped or silently adjusted.
    Depth is not checked against independent hydraulic/geological constraints.
    """
    weights = ranked['geographic_group_weight'].to_numpy(dtype=float)
    pab = float(summary['abandonment_pressure_bar']) if cls == 'DGF' else None
    pmin = float(summary['minimum_pressure_bar'])
    pmax = float(summary['maximum_pressure_bar'])
    rows = []
    for source_col, output_col, unit in PARAMETERS[cls]:
        q = {f'q{k:02d}': weighted_quantile(ranked[source_col], weights, k / 100)
             for k in QUANTILES}
        ref = float(summary[output_col])
        result = dict(storage_class=cls, parameter=source_col, unit=unit,
                      n_rows=len(ranked),
                      n_geographic_groups=ranked.geographic_group.nunique(),
                      reference_value=ref, **q)
        for k in QUANTILES:
            val = q[f'q{k:02d}']
            a, b, c = pab, pmin, pmax
            if output_col == 'abandonment_pressure_bar':
                a = val
            elif output_col == 'minimum_pressure_bar':
                b = val
            elif output_col == 'maximum_pressure_bar':
                c = val
            feasible = (0 < b < c and (a is None or 0 < a < b))
            result[f'q{k:02d}_oat_pressure_feasible'] = bool(feasible)
        rows.append(result)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,default=Path('outputs/cost_sensitivity/reservoir_sample_results.csv'))
    p.add_argument('--groups',type=Path,default=None,
                   help='Reviewed CSV containing storage_class, site, geographic_group')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/reservoir_reference_calibration'))
    p.add_argument('--maxiter',type=int,default=50)
    p.add_argument('--starts',type=int,default=4)
    args=p.parse_args()
    data=pd.read_csv(args.samples,dtype={'source_site_id':str})
    data=attach_groups(data,args.groups)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    parameter_rows=[]
    for cls in ('DGF','SA'):
        print(f'Calibrating {cls}...')
        summary, ranked=fit_one(cls,data,args.maxiter,args.starts)
        summaries.append(summary)
        parameter_rows.extend(parameter_quantile_rows(cls, ranked, summary))
        ranked.to_csv(args.output_dir/f'{cls.lower()}_ranked_sites.csv',index=False)
        print(f"  {summary['n_rows']} complete rows, {summary['n_geographic_groups']} groups")
        print(f"  Nearest observed site: {summary['reference_site']}")
        print(f"  Calibrated depth={summary['depth_m']:.1f}m, "
              f"Pab={summary['abandonment_pressure_bar']:.1f}bar, "
              f"Pmin={summary['minimum_pressure_bar']:.1f}bar, "
              f"Pmax={summary['maximum_pressure_bar']:.1f}bar")
        for cost in COSTS:
            print(f"    {cost}: Q50={summary['q50_'+cost]:.4f} "
                  f"fitted={summary['calibrated_'+cost]:.4f} "
                  f"({summary['error_pct_'+cost]:+.1f}%)")
    pd.DataFrame(summaries).to_csv(args.output_dir/'calibrated_reference_configs.csv',index=False)
    parameter_table = pd.DataFrame(parameter_rows)
    parameter_table.to_csv(args.output_dir/'reservoir_parameter_quantiles.csv', index=False)
    print('\nGeography-weighted parameter quantiles (pressure feasibility at fixed fitted reference):')
    for _, row in parameter_table.iterrows():
        print(f"  {row['storage_class']} {row['parameter']}: "
              f"Q10={row['q10']:.2f}, Q25={row['q25']:.2f}, "
              f"Q50={row['q50']:.2f}, Q75={row['q75']:.2f}, Q90={row['q90']:.2f} "
              f"(reference={row['reference_value']:.2f} {row['unit']})")
        for qlabel in ('q10', 'q25', 'q75', 'q90'):
            if not row[f'{qlabel}_oat_pressure_feasible']:
                print(f"    WARNING: {qlabel.upper()} invalid as an independent pressure OAT "
                      f"against the fitted reference ({row[qlabel]:.2f} {row['unit']})")
    data[['storage_class','site','country','geographic_group']].to_csv(
        args.output_dir/'reviewed_group_memberships.csv',index=False)
    print('Saved:', args.output_dir)


if __name__=='__main__':
    main()
