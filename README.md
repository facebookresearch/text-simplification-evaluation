# Reference-less Quality Estimation of Text Simplification Systems

This repository contains the original implementation of the evaluation methods presented in [Reference-less Quality Estimation of Text Simplification Systems](https://www.aclweb.org/anthology/W18-7005) (1st Workshop on Automatic Text Adaption, INLG 2018).
The version that was used at submission time is on branch [submission](https://github.com/facebookresearch/text-simplification-evaluation/tree/submission).

You can get a quick glance on what the code does in [demo/reference_less_evaluation.ipynb](https://github.com/facebookresearch/text-simplification-evaluation/blob/master/demo/reference_less_evaluation.ipynb).

## Getting Started

### Dependencies

* Python 3.6
* Java
* Optional: [QuEst](https://github.com/ghpaetzold/questplusplus) (install in `resources/tools/quest/`)
* Optional: [TERp](https://github.com/snover/terp) (install in `resources/tools/terp/`)

### Installing

```
git clone git@github.com:facebookresearch/text-simplification-evaluation.git
cd text-simplification-evaluation
pip install -e .
pip install -r requirements.txt
```

## Running the tests
```
pytest tests/
```

## References

If you use this code, please cite:  
L. Martin, S. Humeau, PE. Mazaré, E. De la Clergerie, A. Bordes, B. Sagot, [*Reference-less Quality Estimation of Text Simplification Systems*](https://arxiv.org/abs/1901.10746)

## Author

If you have any question, please contact the author:
**Louis Martin** ([louismartin@fb.com](mailto:louismartin@fb.com))

## License

See the [LICENSE](LICENSE) file for more details.
