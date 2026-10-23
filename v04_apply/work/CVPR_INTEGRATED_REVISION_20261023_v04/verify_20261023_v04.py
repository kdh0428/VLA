"""Independent raw-token CPU review; NumPy and Pillow only. No model or simulator.

Run from any directory: python /path/to/verify_20261023_v04.py
The source records, native decoder config and every bootstrap result are local.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PC = json.loads((ROOT / 'snapshot/ckpt_config/processor_config.json').read_text())
BP = PC['bin_policy']
RANGES = ((257153, 261248), (261249, 265344), (265345, 265346))
N = Counter()


def require(value, label):
    N['checks'] += 1
    if not value: raise AssertionError(label)


def same(x, y, label):
    require(np.allclose(x, y, atol=2e-14, rtol=0), label)


def typed(ids):
    return all(RANGES[j % 3][0] <= int(v) <= RANGES[j % 3][1] for j, v in enumerate(ids))


def center(group, name, idx):
    edge = np.asarray(BP[group][name])
    return (edge[idx] + edge[idx + 1]) / 2


def decode(ids):
    require(len(ids) % 3 == 0 and typed(ids), 'typed token triplets')
    a = np.asarray(ids, dtype=np.int64).reshape(-1, 3)
    t = a[:, 0] - RANGES[0][0]
    theta = center('translation', 'theta_bins', t // 256)
    phi = center('translation', 'phi_bins', t % 256 // 8)
    radius = center('translation', 'r_bins', t % 8)
    xyz = np.column_stack((radius * np.sin(theta) * np.cos(phi),
                           radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)))
    r = a[:, 1] - RANGES[1][0]
    rpy = np.column_stack((center('rotation', 'roll_bins', r // 256),
                           center('rotation', 'pitch_bins', r % 256 // 16),
                           center('rotation', 'yaw_bins', r % 16)))
    return np.column_stack((np.clip(xyz, -1, 1), rpy, (a[:, 2] - RANGES[2][0]).astype(float)))


def d(x, y):
    return float(np.abs(x[:, :6] - y[:, :6]).mean())


METRICS = ['Delta', 'E_L', 'E_O', 'E_OL', 'D_OL_L', 'J', 'E_L_a3', 'E_O_a3', 'E_L_a4',
           'E_O_a4', 'E_L_trans', 'E_O_trans', 'E_L_rot', 'E_O_rot',
           'grip_change_L', 'grip_change_O', 'grip_change_OL']


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows: grouped[(r['task'], r['seed'])].append([r[m] for m in METRICS])
    tasks = sorted({t for t, _ in grouped})
    by = {t: np.asarray([np.mean(grouped[k], axis=0) for k in sorted(grouped) if k[0] == t]) for t in tasks}
    tm = {t: a.mean(axis=0) for t, a in by.items()}
    rng = np.random.default_rng(20261023)
    bt = []; tbt = {t: [] for t in tasks}
    for _ in range(10000):
        for t in tasks:
            x = by[t]
            tbt[t].append(x[rng.integers(0, len(x), len(x))].mean(axis=0))
        bt.append(np.mean([tbt[t][-1] for t in tasks], axis=0))
    result = {}
    for j, m in enumerate(METRICS):
        result[m] = {'estimate': float(np.mean([tm[t][j] for t in tasks])),
                     'ci': (None if min(len(x) for x in by.values()) <= 1 else
                            np.quantile(np.asarray(bt)[:, j], [.025, .975]).tolist()),
                     'n_obs': len(rows), 'n_layouts': {t: len(x) for t, x in by.items()},
                     'task_means': {t: float(tm[t][j]) for t in tasks},
                     'task_ci': {t: (None if len(by[t]) <= 1 else
                                     np.quantile(np.asarray(tbt[t])[:, j], [.025, .975]).tolist()) for t in tasks}}
    return result


def compare(got, expected, label):
    require(got['n_obs'] == expected['n_obs'] and got['n_layouts'] == expected['n_layouts'], label + ' counts')
    same(got['estimate'], expected['estimate'], label + ' estimate')
    if got['ci'] is None: require(expected['ci'] is None, label + ' NA CI')
    else: same(got['ci'], expected['ci'], label + ' CI')
    for t in got['task_means']:
        same(got['task_means'][t], expected['task_means'][t], label + ' task mean')
        if got['task_ci'][t] is None: require(expected['task_ci'][t] is None, label + ' task NA CI')
        else: same(got['task_ci'][t], expected['task_ci'][t], label + ' task CI')


datasets = {}
all_rows = {}
for label, directory, phase in [
    ('followup', 'CVPR_SPATIALVLA_WITHIN_CALL_RESULT_20261023_v02', 'main'),
    ('drawer', 'CVPR_SPATIALVLA_DRAWER_RESULT_20261023_v01', 'recovered')]:
    base = ROOT / directory
    records = [json.loads(l) for l in (base / 'run' / phase / 'records.jsonl').read_text().splitlines()]
    expected = json.loads((base / 'analysis/summary.json').read_text())
    pngs = {p.stem: p for p in base.rglob('*.png')}
    rows = []; rejects = Counter(); counts = Counter()
    for r in records:
        ids = r['official']['ids']
        require(len(ids) == 13 and ids[-1] == 1 and typed(ids[:12]), 'native typed length and EOS')
        require(r['adapter_natural']['ids'] == ids and r['R']['ids'] == ids[6:], 'raw parity and EOS')
        collection = r['official']['equals_collect_predict_action']
        if label == 'followup': require(collection is True, 'mandatory main collection parity')
        else: require(collection is None, 'disclosed drawer collection comparison unavailable')
        require(r['R']['decoded_equal'] is True, 'recorded decoded equality')
        rgb = np.asarray(Image.open(pngs[r['obs_id']]).convert('RGB'))
        require(hashlib.sha256(rgb.tobytes()).hexdigest() == r['img_sha256'], 'raw RGB input hash')
        un = decode(ids[:12]); same(un, r['u_N'], 'native decoded action')
        source = None
        candidates = r['sources']
        require(1 <= len(candidates) <= 64 and len(candidates) == r['n_sources_tried'], 'source count cap')
        for j, s in enumerate(candidates):
            counts['sources'] += 1
            require(s['j'] == j, 'source order')
            seed = int.from_bytes(hashlib.sha256(f"svla_wc_v02_src:{r['obs_id']}:{j}".encode()).digest()[:4], 'little')
            require(s['seed'] == seed, 'source seed')
            a = s['ids'][:6]
            reason = 'invalid_type_or_eos_in_a1a2'
            if len(a) == 6 and typed(a):
                u = decode(a)
                do = float(np.abs(u[0, :6] - un[0, :6]).mean())
                dl = float(np.abs(u[1, :6] - un[1, :6]).mean())
                same([do, dl], [s['d_O'], s['d_L']], 'donor decoded distances')
                if a[2] != ids[2] or a[5] != ids[5]: reason = 'gripper_mismatch'
                elif do == 0 or dl == 0: reason = 'zero_distance'
                elif not .8 <= dl / do <= 1.25: reason = 'ratio_outside_0.8_1.25'
                else: reason = None
            require(reason == s['reason'], 'donor rejection reason')
            rejects[reason or 'selected'] += 1
            if reason is None:
                require(source is None and j == len(candidates) - 1, 'first matched source stopping')
                source = (a, do, dl, j)
        if source is None:
            require(len(candidates) == 64 and r['status'] == 'MATCHED_SOURCE_MISSING', 'exhausted missing source')
            continue
        a, do, dl, j = source
        require(r['donor']['j'] == j and r['status'] == 'OK', 'matched donor and status')
        ur = decode(r['R']['ids'][:6])
        us = {}
        conditions = {'O': a[:3] + ids[3:6], 'L': ids[:3] + a[3:], 'OL': a}
        for k, cond in conditions.items():
            arm = r['arms'][k]
            require(arm['cond'] == cond and arm['emitted_prefix'] == ids[:6], 'conditioning span and emitted prefix')
            require(len(arm['ids']) == 7 and arm['ids'][-1] == 1 and typed(arm['ids'][:6]), 'arm type and EOS')
            us[k] = decode(arm['ids'][:6])
        el, eo, eol = (d(us[k], ur) for k in ('L', 'O', 'OL'))
        row = {'task': r['task'], 'seed': r['seed'], 'obs_id': r['obs_id'], 'd_O': do, 'd_L': dl,
               'Delta': el - eo, 'E_L': el, 'E_O': eo, 'E_OL': eol,
               'D_OL_L': d(us['OL'], us['L']), 'J': eol - eo - el,
               'same_type': (a[0] != ids[0], a[1] != ids[1]) == (a[3] != ids[3], a[4] != ids[4])}
        for k in ('L', 'O'):
            counts['changed_' + k] += r['arms'][k]['ids'][:6] != r['R']['ids'][:6]
            for ti, name in enumerate(('a3', 'a4')):
                row[f'E_{k}_{name}'] = float(np.abs(us[k][ti, :6] - ur[ti, :6]).mean())
            for name, sl in (('trans', slice(0, 3)), ('rot', slice(3, 6))):
                row[f'E_{k}_{name}'] = float(np.abs(us[k][:, sl] - ur[:, sl]).mean())
        for k in ('L', 'O', 'OL'): row['grip_change_' + k] = float((us[k][:, 6] != ur[:, 6]).mean())
        rows.append(row)
    stats = aggregate(rows)
    plotted = {r['obs_id']: r for r in csv.DictReader((base / 'analysis/per_observation.csv').open())}
    require(set(plotted) == {r['obs_id'] for r in rows}, label + ' plotted observation coverage')
    for r in rows:
        for metric in METRICS:
            same(float(plotted[r['obs_id']][metric]), r[metric], label + ' plotted ' + metric)
    compare(stats['Delta'], expected['primary_Delta'], label + ' primary')
    for m in METRICS[1:]: compare(stats[m], expected['secondary'][m], label + ' ' + m)
    pattern = aggregate([r for r in rows if r['same_type']])['Delta']
    compare(pattern, expected['secondary']['Delta_same_change_type'], label + ' same changed-token type')
    require(counts['sources'] == expected['n_sources'] and dict(rejects) == expected['source_reasons'], label + ' sources')
    same([np.mean([r['d_L'] for r in rows]), np.mean([r['d_O'] for r in rows])],
         [expected['donor_distance']['mean_d_L'], expected['donor_distance']['mean_d_O']], label + ' distance means')
    all_rows[label] = rows
    datasets[label] = {'records': len(records), 'primary': stats['Delta'],
                       'source_reasons': dict(rejects), 'source_count': counts['sources'],
                       'token_change_counts': {k: counts['changed_' + k] for k in ('L', 'O')},
                       'same_change_type': pattern,
                       'max_recorded_R_logit_difference': max(r['R']['max_abs_logit_diff_vs_official'] for r in records),
                       'collection_parity_available': label == 'followup'}

# Post-hoc drawer sensitivity: cluster identical recorded poses across target instructions and asset labels.
survey = {r['seed']: r for r in json.loads((ROOT / 'CVPR_SPATIALVLA_DRAWER_RESULT_20261023_v01/manifest/survey_open_drawer_5000_5099.json').read_text())}
scenario = defaultdict(list)
for r in all_rows['drawer']: scenario[r['seed']].append(r['Delta'])
pose = defaultdict(list); members = defaultdict(list)
for seed, values in sorted(scenario.items()):
    s = survey[seed]
    arts = {('cabinet' if k == 'cabinet' else 'robot'): v for k, v in s['articulations'].items()}
    h = hashlib.sha256(json.dumps({'actors': s['actors'], 'articulations': arts}, sort_keys=True).encode()).hexdigest()[:16]
    pose[h].append(float(np.mean(values))); members[h].append(seed)
groups = [np.asarray(pose[k]) for k in sorted(pose)]
rng = np.random.default_rng(20261023); bw = []; be = []
for _ in range(10000):
    chosen = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
    bw.append(np.concatenate(chosen).mean()); be.append(np.mean([g.mean() for g in chosen]))
sensitivity = {'post_hoc': True, 'scenario_count': len(scenario), 'pose_groups': len(groups),
               'scenario_weighted_estimate': float(np.concatenate(groups).mean()),
               'scenario_weighted_pose_cluster_ci': np.quantile(bw, [.025, .975]).tolist(),
               'equal_pose_group_estimate': float(np.mean([g.mean() for g in groups])),
               'equal_pose_group_ci': np.quantile(be, [.025, .975]).tolist(), 'members': dict(members)}
require(sensitivity['pose_groups'] == 7 and sensitivity['scenario_count'] == 15, 'drawer pose grouping')
same(sensitivity['scenario_weighted_estimate'], datasets['drawer']['primary']['estimate'], 'retained scenario-weighted estimand')

ledger = json.loads((ROOT / 'gpu_ledger_v03.json').read_text())
times = {a['phase']: (a['last'] - a['start']) / 60 for a in ledger['attempts']}
result = {'all_checks_passed': True, 'checks': N['checks'], 'datasets': datasets,
          'drawer_geometry_sensitivity': sensitivity, 'recorded_new_gpu_minutes': sum(times.values()),
          'recorded_total_including_discovery_minutes': 20.091789011160532 + sum(times.values()),
          'gpu_independently_rerun': False, 'new_experiment_data_generated': False,
          'scope': 'Raw token, decoded action, input hash, donor rule, aggregation and interval checks. Stored GPU logit summaries are not independently rerun.'}
(ROOT / 'qa/RAW_RECOMPUTATION_20261023_v04.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'all_checks_passed': True, 'checks': N['checks'],
                  'followup': datasets['followup']['primary'], 'drawer': datasets['drawer']['primary'],
                  'drawer_pose_group_sensitivity': sensitivity,
                  'recorded_total_gpu_minutes': result['recorded_total_including_discovery_minutes']}, indent=2))
