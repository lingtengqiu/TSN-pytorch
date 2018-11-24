# TSN-Pytorch
所有的操作我们都写在了do.bash 这个文件里头    
主要有rgb 和rgb diff    
光流的提取，这里使用flownet2.0提取，出了些小小的操作并没有达到论文中的结果    
经过我们的测试RGB 和 RGB diff 的联合分布模型 能够达到 89.32%的精度   

# USING
首先你需要软连接你的weights  ln -s ../weights weights  
另外关于数据集你需要将其放在  ../dataset 文件夹里头  
对于结果，我们将其放在 ./result 文件夹内 最后通过 python test_model 获得最终的模型结果  
python vote --Flow None --RGB True --RGBDiff True 获得最终的投票结果  


