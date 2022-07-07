### 网页内容

共**4**个主要页面，分别为:

- 主页(`Home`)：项目与模型介绍
- 数据展示页(`DataViewer`)：数据集介绍
- 阿尔兹海默症预测页(`ADPrediction`)：阿尔兹海默症介绍与风险预测工具
- 帕金森病预测页(`PDPrediction`)：帕金森病介绍与风险预测工具

### 配置教程

#### 环境需求

##### 后端数据库

- MySQL

##### 网页框架

基本环境：Python3

第三方库：

- joblib==1.0.1
- **Flask**==1.0.2
- Pymysql==1.0.2

##### 模型训练

- sklearn==1.1.1

> 项目文件夹中已经保存了训练好的模型，默认情况下不需要配置



#### 后端数据库搭建

1. 安装MySQL

   在本地安装数据库管理系统MySQL，并配置MySQL命令添加至环境变量

   - [Windows安装配置教程](https://blog.csdn.net/qq_59636442/article/details/123058454)

   - [Mac安装配置教程](https://www.jianshu.com/p/a9ed0e783aab)
   - [Linux(CentOS)安装配置教程](https://blog.csdn.net/xhmico/article/details/125197747)

2. 建立数据库

   运行以下代码进行生成建库语句：

   ```shell
   # Clone from GitHub
   git clone DiseasePageProj
   cd DiseasePageProj
   # 本地文件只需要进入目录，即 cd DiseasePageProj
   python3 create_db.py
   ```

   复制生成的建库语句到MySQL终端：

   ```sql
   source /Users/tianpei/PycharmProjects/DiseasePageProj/DB_SQL/Init.sql;
   source /Users/tianpei/PycharmProjects/DiseasePageProj/DB_SQL/AD.sql;
   source /Users/tianpei/PycharmProjects/DiseasePageProj/DB_SQL/PD.sql;
   ```

3. 检测数据库是否建立成功

   MySQL终端输入：

   ```sql
   select * from ADPred limit 3;
   ```

   输出3行数据，则数据成功导入。

   MySQL终端输入：

   ```sql
   select * from PDAudioPred limit 3;
   ```

   输出3行数据，则数据成功导入。





#### 网页环境配置

1. 第三方库安装：

   ```shell
   pip install -r requirements.txt
   ```

2. 修改配置

   打开config.py文件，修改本地数据库参数。

3. 运行网页服务

   ```shell
   python app.py
   ```

4. 访问网页

   打开任意浏览器，输入`http://127.0.0.1:5000/Home`访问主页



#### 预测工具测试

##### AD风险预测

```python
# 逐个复制至http://127.0.0.1:5000/ADPrediction的输入框中
# 记得选择要使用的模型！
'男'    # 性别 
74     # 年龄 
2     # 受教育程度
3      # 社会经济地位 
29      # MMSE
1344      # eTIV
0.743      # nWBV
1.306     # ASF
```

##### PD风险预测

```python
# 直接复制至http://127.0.0.1:5000/PDPrediction的输入框中
# 记得选择要使用的模型！
# 音频参数：
0.827,7.66E-05,0.464,0.481,1.393,5.056,0.464,2.683,3.032,4.547,8.049,0.119402,9.859,108.015,107.959,102.496,116.847,192,191,0.09263342
```











