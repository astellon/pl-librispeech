# Building the docker container.

import argparse
import subprocess
import yaml

parser = argparse.ArgumentParser('Build and push the docker container.')

parser.add_argument('--config', '-c', type=str, default='config.yaml',
                    help='Configuration file. Settings may be overridden by command line arguments.')
parser.add_argument('--repository', type=str, default=None,
                    help='URI to a repository that the built cointainer will be pushed')
parser.add_argument('--tag', '-t', type=str, default=None,
                    help='Container\'s tag. The built container will be pushed to <repository>/<tag> .')

args = parser.parse_args()

with open(args.config) as f:
    conf = yaml.safe_load(f)
    for k, v in conf.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


tag = args.repository + '/' + args.tag

subprocess.run(['docker', 'build', '.', '-t', tag])
subprocess.run(['docker', 'push', tag])
