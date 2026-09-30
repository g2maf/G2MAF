"""
Standalone converter for OG-MARL SMAC v1 TFRecord datasets to npy format.
Does NOT require og-marl installation or StarCraft2Env.

Usage:
  python convert_smac_data.py --map_name 3m --quality Medium \
    --zip_path /path/to/3m.zip \
    --out_dir /path/to/diffuser/datasets/data/smac
"""
import os
import sys
import shutil
import zipfile
import tempfile
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Known SMAC map dims (obs_size = raw obs without agent_id)
SMAC_MAPS = {
    '3m':       {'n_agents': 3, 'obs_size': 7,  'n_actions': 9,  'episode_limit': 60},
    '5m_vs_6m': {'n_agents': 5, 'obs_size': 72, 'n_actions': 14, 'episode_limit': 70},
    '2s3z':     {'n_agents': 5, 'obs_size': 80, 'n_actions': 16, 'episode_limit': 120},
    '8m':       {'n_agents': 8, 'obs_size': 7,  'n_actions': 14, 'episode_limit': 120},
}

PERIOD = 10  # timesteps per TFRecord entry (og-marl standard)


def build_schema(n_agents):
    schema = {}
    for i in range(n_agents):
        agent = 'agent_{}'.format(i)
        schema[agent + '_observations'] = tf.io.FixedLenFeature([], dtype=tf.string)
        schema[agent + '_legal_actions'] = tf.io.FixedLenFeature([], dtype=tf.string)
        schema[agent + '_actions']       = tf.io.FixedLenFeature([], dtype=tf.string)
        schema[agent + '_rewards']       = tf.io.FixedLenFeature([], dtype=tf.string)
        schema[agent + '_discounts']     = tf.io.FixedLenFeature([], dtype=tf.string)
    schema['zero_padding_mask'] = tf.io.FixedLenFeature([], dtype=tf.string)
    schema['env_state']         = tf.io.FixedLenFeature([], dtype=tf.string)
    schema['episode_return']    = tf.io.FixedLenFeature([], dtype=tf.string)
    return schema


def decode_record(record_bytes, schema, n_agents):
    example = tf.io.parse_single_example(record_bytes, schema)
    result = {}
    for i in range(n_agents):
        agent = 'agent_{}'.format(i)
        result[agent] = {
            'obs':   tf.io.parse_tensor(example[agent + '_observations'], tf.float32),
            'legal': tf.io.parse_tensor(example[agent + '_legal_actions'], tf.float32),
            'act':   tf.io.parse_tensor(example[agent + '_actions'],       tf.float32),
            'rew':   tf.io.parse_tensor(example[agent + '_rewards'],       tf.float32),
            'disc':  tf.io.parse_tensor(example[agent + '_discounts'],     tf.float32),
        }
    result['zero_padding_mask'] = tf.io.parse_tensor(example['zero_padding_mask'], tf.float32)
    try:
        result['env_state'] = tf.io.parse_tensor(example['env_state'], tf.float32)
    except Exception:
        result['env_state'] = tf.zeros([PERIOD, 1])
    return result


def convert(zip_path, map_name, quality, out_dir):
    info = SMAC_MAPS[map_name]
    n_agents  = info['n_agents']
    obs_size  = info['obs_size']
    n_actions = info['n_actions']
    ep_limit  = info['episode_limit']

    out_path = os.path.join(out_dir, map_name, quality)
    os.makedirs(out_path, exist_ok=True)

    done_files = ['obs.npy', 'actions.npy', 'rewards.npy', 'legals.npy', 'path_lengths.npy']
    if all(os.path.exists(os.path.join(out_path, f)) for f in done_files):
        print("[SKIP] Already converted: {}".format(out_path))
        return True

    print("Converting {}-{} (n_agents={}, obs_size={}, n_actions={})".format(
        map_name, quality, n_agents, obs_size, n_actions))

    tmp_dir = tempfile.mkdtemp(prefix='smac_')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = zf.namelist()
            tffiles = sorted([n for n in all_names
                              if quality in n and n.endswith('.tfrecord')])
            if not tffiles:
                tffiles = sorted([n for n in all_names if n.endswith('.tfrecord')])
                tffiles = [n for n in tffiles if quality.lower() in n.lower()]
            if not tffiles:
                print("Available entries (first 20):", all_names[:20])
                raise ValueError("No TFRecord files for quality={}".format(quality))

            print("Extracting {} TFRecord files...".format(len(tffiles)))
            for fname in tffiles:
                zf.extract(fname, tmp_dir)

        full_paths = [os.path.join(tmp_dir, f) for f in tffiles
                      if os.path.exists(os.path.join(tmp_dir, f))]
        if not full_paths:
            raise ValueError("Extraction failed - no files found in tmp_dir")

        print("Decoding {} files...".format(len(full_paths)))
        schema = build_schema(n_agents)
        dataset = tf.data.TFRecordDataset(full_paths, compression_type='GZIP')

        all_obs    = []
        all_acts   = []
        all_rews   = []
        all_legals = []
        all_path_lengths = []

        path_obs, path_acts, path_rews, path_legals = [], [], [], []
        path_length = 0
        n_episodes = 0

        for record_bytes in dataset:
            try:
                rec = decode_record(record_bytes, schema, n_agents)
            except Exception as e:
                sys.stderr.write("Warning decode: {}\n".format(e))
                continue

            zp = rec['zero_padding_mask'].numpy()
            valid = int(np.sum(zp))
            if valid == 0:
                continue

            obs_s   = np.stack([rec['agent_{}'.format(i)]['obs'].numpy()
                                 for i in range(n_agents)], axis=1)  # [P, N, obs]
            act_s   = np.stack([rec['agent_{}'.format(i)]['act'].numpy()
                                 for i in range(n_agents)], axis=1)  # [P, N, ...]
            rew_s   = np.stack([rec['agent_{}'.format(i)]['rew'].numpy()
                                 for i in range(n_agents)], axis=1)  # [P, N]
            disc_s  = np.stack([rec['agent_{}'.format(i)]['disc'].numpy()
                                 for i in range(n_agents)], axis=1)
            legal_s = np.stack([rec['agent_{}'.format(i)]['legal'].numpy()
                                  for i in range(n_agents)], axis=1) # [P, N, A]

            path_obs.append(obs_s[:valid])
            path_acts.append(act_s[:valid])
            path_rews.append(rew_s[:valid])
            path_legals.append(legal_s[:valid])
            path_length += valid

            # Episode ends when discount == 0 or hit episode_limit
            terminal = (disc_s[valid - 1, 0] < 0.5)
            if terminal or path_length >= ep_limit:
                p_obs   = np.concatenate(path_obs, axis=0)
                p_acts  = np.concatenate(path_acts, axis=0)
                p_rews  = np.concatenate(path_rews, axis=0)
                p_legals = np.concatenate(path_legals, axis=0)

                T = p_obs.shape[0]
                # Add one-hot agent IDs
                ids = np.tile(np.eye(n_agents, dtype=np.float32)[None],
                              [T, 1, 1])  # [T, N, N]
                p_obs_id = np.concatenate([ids, p_obs], axis=-1)  # [T, N, N+obs]

                # Flatten actions to integer indices
                if p_acts.ndim == 3 and p_acts.shape[-1] == 1:
                    p_acts = p_acts.squeeze(-1)
                elif p_acts.ndim == 3 and p_acts.shape[-1] > 1:
                    p_acts = np.argmax(p_acts, axis=-1)

                all_obs.append(p_obs_id)
                all_acts.append(p_acts)
                all_rews.append(p_rews)
                all_legals.append(p_legals)
                all_path_lengths.append(path_length)

                path_obs, path_acts, path_rews, path_legals = [], [], [], []
                path_length = 0
                n_episodes += 1

                if n_episodes % 500 == 0:
                    print("  {} episodes decoded...".format(n_episodes))

        if not all_obs:
            raise ValueError("No episodes decoded!")

        print("Total episodes: {}".format(len(all_obs)))

        concat_obs    = np.concatenate(all_obs, axis=0)
        concat_acts   = np.concatenate(all_acts, axis=0)
        concat_rews   = np.concatenate(all_rews, axis=0)
        concat_legals = np.concatenate(all_legals, axis=0)
        concat_lens   = np.array(all_path_lengths)

        print("obs:    {}".format(concat_obs.shape))
        print("acts:   {}".format(concat_acts.shape))
        print("rews:   {}".format(concat_rews.shape))
        print("legals: {}".format(concat_legals.shape))

        np.save(os.path.join(out_path, 'obs.npy'),          concat_obs)
        np.save(os.path.join(out_path, 'actions.npy'),      concat_acts)
        np.save(os.path.join(out_path, 'rewards.npy'),      concat_rews)
        np.save(os.path.join(out_path, 'legals.npy'),       concat_legals)
        np.save(os.path.join(out_path, 'path_lengths.npy'), concat_lens)

        print("Saved to: {}".format(out_path))
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', required=True, choices=list(SMAC_MAPS.keys()))
    parser.add_argument('--quality', required=True)
    parser.add_argument('--zip_path', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    ok = convert(args.zip_path, args.map_name, args.quality, args.out_dir)
    sys.exit(0 if ok else 1)
