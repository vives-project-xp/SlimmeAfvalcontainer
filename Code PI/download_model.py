from pathlib import Path, PurePosixPath

import requests

TARGET_DIR = Path("/home/kobe/smart_bin")
GITHUB_OWNER = "vives-project-xp"
GITHUB_REPO = "SlimmeAfvalcontainer"
GITHUB_BRANCH = "main"
GITHUB_SOURCE_DIR = "Code PI"


def github_contents_url(path: str) -> str:
    return (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}"
        f"?ref={GITHUB_BRANCH}"
    )


def list_github_files(path: str):
    response = requests.get(github_contents_url(path), timeout=30)
    response.raise_for_status()
    items = response.json()

    files = []
    for item in items:
        if item["type"] == "file":
            files.append(item)
        elif item["type"] == "dir":
            files.extend(list_github_files(item["path"]))
    return files


def write_file_if_changed(destination: Path, content: bytes) -> str:
    if destination.exists() and destination.read_bytes() == content:
        return "ongewijzigd"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return "bijgewerkt" if destination.exists() else "gedownload"


def sync_github_code_pi_to_target():
    """Download alle bestanden uit GitHub map 'Code PI' naar /home/kobe/smart_bin."""
    print("=" * 60)
    print(f"Syncing GitHub '{GITHUB_SOURCE_DIR}' naar {TARGET_DIR}...")
    print("=" * 60)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    files = list_github_files(GITHUB_SOURCE_DIR)
    print(f"Gevonden op GitHub: {len(files)} bestanden")

    downloaded_count = 0
    unchanged_count = 0
    failed_count = 0

    for item in files:
        source_path = PurePosixPath(item["path"])
        relative_path = source_path.relative_to(GITHUB_SOURCE_DIR)
        destination = TARGET_DIR / Path(relative_path.as_posix())

        try:
            file_response = requests.get(item["download_url"], timeout=30)
            file_response.raise_for_status()

            existed_before = destination.exists()
            status = write_file_if_changed(destination, file_response.content)

            if status == "ongewijzigd":
                unchanged_count += 1
                print(f"= Ongewijzigd: {relative_path}")
            else:
                downloaded_count += 1
                action = "Bijgewerkt" if existed_before else "Gedownload"
                print(f"✓ {action}: {relative_path}")

        except Exception as error:
            failed_count += 1
            print(f"✗ Fout bij {relative_path}: {error}")

    print("\n" + "=" * 60)
    print("Klaar met synchroniseren")
    print(f"✓ Nieuw/bijgewerkt: {downloaded_count}")
    print(f"= Ongewijzigd: {unchanged_count}")
    print(f"✗ Mislukt: {failed_count}")
    print(f"Doelmap: {TARGET_DIR}")

    if (TARGET_DIR / "model.onnx").exists() and (TARGET_DIR / "model.onnx.data").exists():
        print("✓ Modelbestanden gevonden in doelmap.")
    else:
        print("⚠ Let op: model.onnx en/of model.onnx.data ontbreken in GitHub map Code PI.")
    print("=" * 60)


if __name__ == "__main__":
    sync_github_code_pi_to_target()
