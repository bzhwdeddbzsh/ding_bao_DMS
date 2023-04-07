# Ding_Bao_DMS
# 这是一个简单的不太成熟的DMS系统他叫盯宝
## 先介绍一下他的来源：
首先主要模型来源于Effcientnet（https://arxiv.org/abs/1905.11946） 
其次大部分的代码实现来源于（https://github.com/WZMIAOMIAO/deep-learning-for-image-processing）
还有一部分视线检测的模块实现来源于MobileFaceNet+MCTNN,但是目前我们忘记了它的原作者（这里欢迎原作者找我们修改，或者有知道的大佬可以通知我们）
以上可以看出盯宝是个究极缝合怪，所以十分感谢各位大佬的帮助
## 接下来介绍他的功能：
1. 我们用CBAM+Effcientnet完成了主体的行为监测
2. 并且用MobileFaceNet+MCTNN进行关键点监测以达到监控视线的功能
## 然后是他的使用结果
前面提到了它不太成熟，所以正确率不是很高，但是起码能用，所以希望用过它的人或者各位大佬能给我们提点意见
## 最后是他的使用方法
1. 下载所有文件并给我们一个star（不给的话打不开哟）需要安装的包见文件中的requirement.txt，还得配好CUDA
2. 打开MAIN，里面有简单注释
3. 调整好相关地址，运行MAIN
4. 你就获得了我们可爱的盯宝，请好好对她
5. ps：因为真的很仓促，所以注释不是很全面，我们会慢慢完善，可能不是特别适合新手小白对他进行修改。
## 最后的最后
因为我们参加了商业比赛，所以它已经被我们注册成软著了（虽然没什么用）但是如果要用于商业的话请与我们联系，我们秉承着开放包容的态度面向全部人

