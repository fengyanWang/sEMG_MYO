文件说明：
	FeaturesFcn：一些特征提取函数
	FeatureSpace：调用上面特征提取函数，生成特征空间
	myClf：模型的离线训练
	onLineClf：模型的在线训练
	open_myo：从传感器上读取数据
	saveMyoData：存储肌电和姿态数据（csv和pickle）
环境说明：
	树莓派3 + 官方的操作系统 + Python2.7 + sklearn16

其他说明：
	所有的文件采用的都是相对路径进行保存的，所以可以很方便的移植到其他的平台上去
	