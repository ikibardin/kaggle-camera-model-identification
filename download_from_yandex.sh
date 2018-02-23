#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

pushd downloader
pushd yandex
python download_from_yandex.py
popd
popd