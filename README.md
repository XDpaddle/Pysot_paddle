# PySOT

**PySOT** 是由SenseTime视频智能研究团队设计的软件系统，包含 [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) 与 [SiamMask](https://arxiv.org/abs/1812.05050)。 

PySOT支持以下项目: [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html), [DaSiamRPN](https://arxiv.org/abs/1808.06048), [SiamRPN++](https://arxiv.org/abs/1812.11703), 和 [SiamMask](https://arxiv.org/abs/1812.05050).

<div align="center">
  <img src="demo/output/bag_demo.gif" width="800px" />
  <p>Example SiamFC, SiamRPN and SiamMask outputs.</p>
</div>

### 安装

- 安装paddle。
- 安装paddleseg，输入pip install paddleseg
- 进入工程文件根目录，输入pip install -r requirements.txt
- python setup.py build_ext --inplace

## 以下文件位于tools目录中运行前应在命令行中输入

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

### Demo
运行demo.py。通过更改config,snapshot,video_name，三个参数，改变模型的配置文件，权重文件，测试视频名称。Demo视频路径demo，输出结果路径demo/output。


### Test
运行test.py，对指定的测试集进行测试。config,snapshot两参数与demo中相同，dataset为选用的测试集，video测试测试集中指定视频。vis是否显示结果。测试集标定结果默认保存路径results。



### Eval
运行eval.py。对test标定结果进行验证。tracker_path，跟踪器标定结果路径，dataset，测试集名称。num，进行eval的进程数tracker_prefix，存储标定结果的文件夹名称（路径），总路径为results/dataset/tracker_prefix。show_video_level显示每个视频测试结果。

### Train
运行train_single.py。cfg参数为训练配置文件。

## 参考：
本项目代码参考自：https://github.com/STVIR/pysot
SiamRPN paperwithcode：https://paperswithcode.com/paper/high-performance-visual-tracking-with-siamese
SiamRPN++ paperwithcode：https://paperswithcode.com/paper/siamrpn-evolution-of-siamese-visual-tracking
SiamMask paperwithcode：https://paperswithcode.com/paper/siammask-a-framework-for-fast-online-object
