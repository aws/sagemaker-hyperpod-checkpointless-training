import subprocess
from datetime import datetime

VERSION_PREFIX = "1.0"  # Expected to be formatted like `major.minor`
SEQUENCE_NAME = "hyperpod_engines_faraday_sequence"


def _get_version() -> str:
    """
    Package version based on a version-set sequence, i.e. `major.minor.build_id`. See
    https://docs.hub.amazon.dev/brazil/peru-user-guide/versioning-in-peru/#python
    """
    local = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        result = subprocess.run(
            [
                "brazil-context",
                "package",
                "build-id",
                "--sequence",
                SEQUENCE_NAME,
                "--local-value",
                local,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        print(f"Got exception while running brazil-context to get version, {e}")
        raise
    build_id = result.stdout.strip()

    version = f"{VERSION_PREFIX}.{build_id}"
    print(f"version={version}")
    return version


# With each commit we want to publish a new version of the library to code-artifact. The transform
# that publishes libraries to code-artifact will ignore pushes for a version that is already
# available in the repository and pushing the same version will be treated as no-op. See
# https://builderhub.corp.amazon.com/docs/brazil/peru-user-guide/versioning-in-peru.html.
__version__ = _get_version()
