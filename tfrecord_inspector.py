import re
import argparse
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecord_file")
    args = parser.parse_args()

    for data in tf.python_io.tf_record_iterator(args.tfrecord_file):
        features = tf.train.Example.FromString(data).features
        print("**************")
        for index in re.finditer("feature", str(features)):
            start = index.span()[0]
            end = start + 200
            print(str(features)[start:end])
            print("**************")
        break
