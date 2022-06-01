# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This file should be ran on the target embedded device
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import os
import pickle
from tqdm import tqdm

base_tflite_file = "vgg16.tflite"
pruned_tflite_file = "vgg16-pruned.tflite"

tflite_file = base_tflite_file

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # scale, zero_point = output_details['quantization']
  # output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]][0]


def evaluate_model(tflite_file):
  interpreter = Interpreter(model_path=tflite_file)
  print("Model Loaded Successfully.")

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print("Image Shape (", width, ",", height, ")")

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Iterate over each class folder and the images inside
  bench_data = []
  test_folder = "data/test"

  for subdir, _, files in os.walk(test_folder):
    if subdir == test_folder: continue

    label = subdir.replace(test_folder+"/", "", 1)

    print("Testing images of type:", label)
    with tqdm(total=len(files), ncols=64) as pbar:
      image_count = 0
      image_class_folder = test_folder + "/" + label

      for filename in os.listdir(image_class_folder):
        image = Image.open(image_class_folder + "/" + filename).convert('RGB').resize((width, height))
        benchmark = {}
        benchmark["label"] = label

        # Classify the image.
        time1 = time.time()
        label_id, prob = classify_image(interpreter, image)
        time2 = time.time()
        classification_time = np.round(time2-time1, 5)
        benchmark["time"] = classification_time
        # print("Classification Time =", classification_time, "seconds.")

        # Read class labels.
        labels = ["paper", "rock", "scissors"]

        # Return the classification label of the image.
        classification_label = labels[label_id]
        confidence = np.round(prob*100, 2)
        benchmark["confidence"] = confidence
        benchmark["correctness"] = classification_label == label

        # print("Image Label is :", classification_label, ", with Accuracy :", accuracy, "%.")

        bench_data.append(benchmark)
        image_count += 1

        # print(f"{class_count * 10 + image_count}/1000")
        pbar.update(1)

        # if image_count == 10: # TEMPORARY
          # break

  with open(tflite_file+"-benchmark.p", 'wb') as f:
    pickle.dump(bench_data, f)
    print("Wrote benchmark data as pickled file.")

def analyze_bench(tflite_file):
  print("--- ANALYZING", tflite_file)

  with open(tflite_file+"-benchmark.p", 'rb') as f:
    benchmark = pickle.load(f)

  total_time = sum(float(item["time"]) for item in benchmark)
  total_wrong = sum(1 if item["correctness"] else 0 for item in benchmark)

  print("Average inference time:", round(total_time/len(benchmark) * 1000, 2), "ms")
  print("Accuracy: {}%".format(round(total_wrong/len(benchmark) * 100, 2)))


def main():
  should_evaluate = True

  if should_evaluate:
    evaluate_model(base_tflite_file)
    evaluate_model(pruned_tflite_file)

  analyze_bench(base_tflite_file)
  print()
  analyze_bench(pruned_tflite_file)

if __name__ == "__main__":
  main()