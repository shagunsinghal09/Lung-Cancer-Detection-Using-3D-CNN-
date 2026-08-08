import argparse
import json

from ml.pipeline import LungCancerPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_path", help="Path to .npy/.nii/.nii.gz 3D CT volume")
    args = parser.parse_args()

    pipeline = LungCancerPipeline()
    result = pipeline.predict(args.scan_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
