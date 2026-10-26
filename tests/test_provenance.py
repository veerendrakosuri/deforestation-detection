import tempfile
import unittest
from pathlib import Path

from deforestation_detection.config import load_config
from deforestation_detection.provenance import sha256_file, workflow_provenance


class ProvenanceTests(unittest.TestCase):
    def test_hashes_and_records_inputs(self) -> None:
        root = Path("outputs/test-temporary")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            source = Path(directory) / "input.bin"
            source.write_bytes(b"forest-change")

            provenance = workflow_provenance(
                load_config(),
                config_path=None,
                inputs={"imagery": source},
                hash_inputs=True,
            )

            self.assertEqual(
                provenance["inputs"]["imagery"]["sha256"],
                sha256_file(source),
            )
            self.assertEqual(provenance["inputs"]["imagery"]["size_bytes"], 13)
            self.assertTrue(provenance["inputs_hashed"])


if __name__ == "__main__":
    unittest.main()
