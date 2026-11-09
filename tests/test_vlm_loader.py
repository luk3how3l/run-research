"""Loader behavior with fake dependencies: no model construction or downloads."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class Loader(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.is_bf16_supported.return_value = True
        self.smol = Mock()
        self.pixtral = Mock()
        self.config = Mock()
        self.processor = Mock()
        self.quantizer = Mock()
        transformers = SimpleNamespace(AutoConfig=self.config, AutoProcessor=self.processor,
                                       Idefics3ForConditionalGeneration=self.smol,
                                       LlavaForConditionalGeneration=self.pixtral,
                                       BitsAndBytesConfig=self.quantizer)
        patches = patch.dict(sys.modules, {'torch': self.torch, 'numpy': Mock(),
                            'requests': Mock(), 'PIL': SimpleNamespace(Image=Mock()),
                            'cv2': Mock(), 'transformers': transformers})
        patches.start()
        self.addCleanup(patches.stop)
        spec = importlib.util.spec_from_file_location('loader_under_test', ROOT / 'src/vlm_vision.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_cpu_smol_offline_revision(self):
        self.config.from_pretrained.return_value = SimpleNamespace(model_type='idefics3')
        self.module.get_model('local-dir', device='cpu', revision='a'*40)
        options = self.smol.from_pretrained.call_args.kwargs
        self.assertTrue(options['local_files_only'])
        self.assertFalse(options['trust_remote_code'])
        self.assertTrue(options['use_safetensors'])
        self.assertEqual(options['revision'], 'a'*40)
        self.assertNotIn('device_map', options)
        self.pixtral.from_pretrained.assert_not_called()
        self.processor.from_pretrained.assert_called_once_with('local-dir', local_files_only=True,
                                                              trust_remote_code=False, revision='a'*40)

    def test_pixtral_nf4_one_logical_gpu(self):
        self.config.from_pretrained.return_value = SimpleNamespace(model_type='llava',
                                                    vision_config=SimpleNamespace(model_type='pixtral'))
        self.module.get_model('pixtral-local', device='cuda', quantization='nf4')
        options = self.pixtral.from_pretrained.call_args.kwargs
        self.assertEqual(options['device_map'], {'': 0})
        self.assertIs(options['quantization_config'], self.quantizer.return_value)
        self.smol.from_pretrained.assert_not_called()

    def test_cuda_unavailable_refuses_before_config(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
            self.module.get_model('local-dir', device='cuda')
        self.config.from_pretrained.assert_not_called()

    def test_cpu_quantization_and_unsupported_architecture(self):
        with self.assertRaises(ValueError):
            self.module.get_model('local-dir', device='cpu', quantization='nf4')
        self.config.from_pretrained.return_value = SimpleNamespace(model_type='qwen')
        with self.assertRaisesRegex(ValueError, 'Unsupported architecture'):
            self.module.get_model('local-dir', device='cpu')
        self.smol.from_pretrained.assert_not_called()
        self.pixtral.from_pretrained.assert_not_called()

    def test_bf16_refusal(self):
        self.torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'BF16'):
            self.module.get_model('local-dir', device='cuda')
        self.config.from_pretrained.assert_not_called()


class CLI(unittest.TestCase):
    def test_bad_cli_fails_without_ml(self):
        for args in [ ['--cpu', '--device', 'cuda'], ['--threads', '0'],
                      ['--revision', 'main'], ['--cpu', '--quantization', 'nf4'] ]:
            r = subprocess.run([sys.executable, '-m', 'start', *args], cwd=ROOT,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertNotIn('ModuleNotFoundError', r.stderr)


if __name__ == '__main__':
    unittest.main()
