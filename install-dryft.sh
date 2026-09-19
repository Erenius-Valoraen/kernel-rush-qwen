#!/bin/sh
set -eu

version=${DRYFT_CLI_VERSION:-0.1.0}
script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
install_dir=${DRYFT_CLI_INSTALL_DIR:-"$script_dir/bin"}

case $(uname -s) in
    Darwin) platform=darwin ;;
    Linux) platform=linux ;;
    *) echo "dryft: unsupported operating system: $(uname -s)" >&2; exit 1 ;;
esac

case $(uname -m) in
    x86_64|amd64) architecture=x86_64 ;;
    arm64|aarch64) architecture=arm64 ;;
    *) echo "dryft: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [ "$platform-$architecture" = "linux-arm64" ]; then
    echo "dryft: Linux arm64 binaries are not published yet" >&2
    exit 1
fi

asset="dryft-$platform-$architecture.tar.gz"
base_url=${DRYFT_CLI_BASE_URL:-"https://dryft-ai.github.io/kernel-deployment-htn-2026/releases/dryft-cli-v$version"}
case "$base_url" in
    https://*) ;;
    http://*)
        if [ "${DRYFT_CLI_ALLOW_INSECURE:-}" != "1" ]; then
            echo "dryft: refusing an insecure download; use an HTTPS release URL" >&2
            exit 1
        fi
        ;;
    *) echo "dryft: release URL must use HTTPS" >&2; exit 1 ;;
esac

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/dryft-install.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM
archive="$tmp_dir/$asset"
checksum="$archive.sha256"

download() {
    url=$1
    output=$2
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error "$url" --output "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet "$url" --output-document="$output"
    else
        echo "dryft: curl or wget is required" >&2
        exit 1
    fi
}

echo "Downloading dryft $version for $platform-$architecture..."
download "$base_url/$asset" "$archive"
download "$base_url/$asset.sha256" "$checksum"

expected=$(awk 'NR == 1 { print $1 }' "$checksum")
case "$expected" in
    *[!0-9a-fA-F]*|"")
        echo "dryft: release checksum is invalid" >&2
        exit 1
        ;;
esac
if [ "${#expected}" -ne 64 ]; then
    echo "dryft: release checksum is invalid" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$archive" | awk '{ print $1 }')
elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$archive" | awk '{ print $1 }')
elif command -v openssl >/dev/null 2>&1; then
    actual=$(openssl dgst -sha256 "$archive" | awk '{ print $NF }')
else
    echo "dryft: sha256sum, shasum, or openssl is required" >&2
    exit 1
fi

if [ "$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')" != \
     "$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')" ]; then
    echo "dryft: checksum verification failed" >&2
    exit 1
fi

mkdir "$tmp_dir/unpacked"
tar -xzf "$archive" -C "$tmp_dir/unpacked"
if [ ! -f "$tmp_dir/unpacked/dryft" ]; then
    echo "dryft: release archive does not contain the dryft executable" >&2
    exit 1
fi

mkdir -p "$install_dir"
if command -v install >/dev/null 2>&1; then
    install -m 0755 "$tmp_dir/unpacked/dryft" "$install_dir/dryft"
else
    cp "$tmp_dir/unpacked/dryft" "$install_dir/dryft"
    chmod 0755 "$install_dir/dryft"
fi

"$install_dir/dryft" version
echo "Installed $install_dir/dryft"
