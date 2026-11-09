"""Run allocated-GPU pipeline checks on RENN. Never falls back to CPU."""
from tools.pipeline_check import main

if __name__ == '__main__':
    main('renn')
