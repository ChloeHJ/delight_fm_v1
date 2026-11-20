"""Launcher and scheduler contracts without site modules, Slurm or a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import pipeline


class LauncherTests(unittest.TestCase):
    def test_interpreters_and_external_run_work_without_site_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'external run'
            run.mkdir()
            interpreter = Path(directory) / 'fake python'
            interpreter.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'print(json.dumps(dict(args=sys.argv[1:], '
                'run=os.environ["DELIGHT_FM_RUN_DIR"], '
                'kind=os.environ.get("MPLBACKEND"))))\n'
            )
            interpreter.chmod(0o700)
            env = dict(os.environ, PYTHON_BIN=str(interpreter),
                       REPORT_PYTHON_BIN=str(interpreter),
                       DELIGHT_FM_RUN_DIR=str(run), DELIGHT_FM_GIT_ROOT=str(ROOT))
            for script, stage in [('run.sh', 'plan'), ('run.sh', 'report'),
                                  ('run.sh', 'cluster'), ('run.sh', 'rebuild-inputs'),
                                  ('job.sh', 'plan')]:
                with self.subTest(script=script, stage=stage):
                    result = subprocess.run(
                        ['bash', '-c', 'module() { return 97; }; export -f module; exec bash "$@"',
                         'test', str(ROOT / 'runs' / script), stage],
                        env=env, text=True, capture_output=True, check=True)
                    value = json.loads(result.stdout)
                    self.assertEqual(value['args'], [str(SRC / 'pipeline.py'), stage])
                    self.assertEqual(value['run'], str(run))
                    self.assertEqual(value['kind'], 'Agg' if stage in ['report', 'cluster']
                                     else env.get('MPLBACKEND'))


class SchedulerTests(unittest.TestCase):
    def commands(self, cpu_partition, gpu_partition):
        settings = dict(slurm=dict(cpu_partition=cpu_partition, gpu_partition=gpu_partition,
                                   time='01:00:00', cpus=2, memory='8G',
                                   training_concurrency=1, evaluation_concurrency=1))
        training = SimpleNamespace(conditions=lambda: [dict(group='immune')], EPOCHS=[0.5])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pipeline, 'RUN', Path(directory)), \
                 patch.object(pipeline, 'config', return_value=settings), \
                 patch.object(pipeline, 'require_plan'), \
                 patch.dict(sys.modules, training=training), \
                 patch('builtins.print') as output:
                pipeline.submit('training', dry_run=True)
            import shlex
            return [shlex.split(call.args[0]) for call in output.call_args_list]

    def test_default_partitions_are_omitted_and_checkout_launcher_is_used(self):
        commands = self.commands(None, None)
        self.assertEqual(len(commands), 6)
        for command in commands:
            self.assertNotIn('--partition', command)
            self.assertEqual(command[-2], str(ROOT / 'runs/job.sh'))

    def test_user_partitions_are_used_for_the_correct_resources(self):
        for command in self.commands('my-cpu', 'my-gpu'):
            expected = 'my-gpu' if '--gpus=1' in command else 'my-cpu'
            self.assertEqual(command[command.index('--partition') + 1], expected)


if __name__ == '__main__':
    unittest.main()
