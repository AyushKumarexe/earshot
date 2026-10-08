#!/usr/bin/env bash
# DCASE 2020 Task 2 dev set (fan) is enough to start. The raw MIMII SNR sets are only needed for E2.
set -euo pipefail
mkdir -p data && cd data

if [ ! -d dev_data/fan ]; then
  curl -L -o dev_data_fan.zip "https://zenodo.org/record/3678171/files/dev_data_fan.zip?download=1"
  mkdir -p dev_data && unzip -q dev_data_fan.zip -d dev_data && rm dev_data_fan.zip
fi

if [ "${1:-}" = "--mimii" ]; then
  mkdir -p mimii
  for snr in 6 0 -6; do
    if [ ! -d "mimii/${snr}_dB_fan" ]; then
      curl -L -o "${snr}_dB_fan.zip" "https://zenodo.org/record/3384388/files/${snr}_dB_fan.zip?download=1"
      mkdir -p "mimii/${snr}_dB_fan" && unzip -q "${snr}_dB_fan.zip" -d "mimii/${snr}_dB_fan" && rm "${snr}_dB_fan.zip"
    fi
  done
fi
ls -d dev_data/fan mimii/* 2>/dev/null
