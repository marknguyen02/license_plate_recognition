# License Plate Recognition

A deep learning system for automatic license plate detection and recognition.

---

## Overview

This project applies **Ultralytics YOLO** for license plate detection and a **PyTorch-based OCR model** for character recognition, supported by **OpenCV** and **NumPy**.

* **Performance**: Achieved **100% accuracy** on the test set of 400 samples, with all license plates correctly recognized.
* **Evaluation**: Extensive visualizations and detailed analysis are available in the data directory.

---

## Workflow
![](./assets/workflow.png)

---

## Results

Comprehensive evaluation outputs are available in [`dataset_test/outputs`](./dataset_test/outputs).  
Real-world test videos are available in [`videos`](./videos/), showcasing system performance under various conditions.

Below are a few illustrative examples:

| Input Image | Model Output |
|-------------|--------------|
| ![Sample 1](./dataset_test/images/carlong_0283.png) | ![Result](./dataset_test/outputs/carlong_0283.png) |
| ![Sample 2](./dataset_test/images/greenpack_1210.png) | ![Result](./dataset_test/outputs/greenpack_1210.png) |

---


## Usage

```bash
git clone https://github.com/marknguyen02/license-plate-recognition.git
cd license-plate-recognition
pip install -r requirements.txt
```

Run the demo notebook demo.ipynb

---

## Notes

* Current version is optimized for **Vietnamese license plates**, extensible to other formats.
* For access to the full dataset or detailed methodology, please contact me.

---

## Contact

- Gmail: [dungnguyen.workspace@gmail.com](mailto:dungnguyen.workspace@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/marknguyen02/](https://www.linkedin.com/in/marknguyen02/)

---

## License

Released under the MIT License.


