# Data-Pre-Processing
Titanic Data-Pre-processing
本次旨在从Titanic Survival Prediction实践中使用数据预处理技术进行比较分析总结

知识背景：数据预处理技术有4种：数据清洗（Data Cleaning）、数据集成（Data Integraiton）、数据归约（Data Reduction）、数据变换（Data Transformation），对应不同情景，可使用每种技术下不同方法，我们将在后文进行分析。

Titanic Survival Prediction实践有两份文件：train.csv和test.csv，数据源：kaggle

shape of train_df (891, 12)
shape of test_df (418, 11)
目标确定：粗略观察数据集差异（train_df提供是否存活，而test_df未），可知此次ML实践旨在通过训练集构建学习模型，来帮助预测测试集的存活情况

一.数据探索(EDA)
读取csv文件，观察train_df变量(由于train数据用于构建模型，test用来测试，故只需对train进行数据探索)：
（一）X：
PassengerId离散型数据，仅用于个体（唯一标识符），无分析意义，在建模中需删除
Pclass有序型数据：类别：1（头等舱）、2（二等舱）、3（三等舱）
Sex、Embarked（登船港口）、Ticket、Cabin：名义型数据，其中Ticket（船票编号）为高基数且对分析无实际作用需删除，Cabin缺失率高，看后文处理
Name文本型数据，需提取称呼（Miss、Mr）作为新特征
Age、Fare连续型数据，Age需处理缺失值，Fare需处理异常值，看后文处理
SibSp（同代直系亲属人数）、Parch（不同代直系亲属人数）：离散型数据
（二）y（目标变量）：Survived二分类目标变量

二.数据清洗
（一）缺失值处理
缺失值类型判断→缺失值处理
缺失值类型：MCAR、MAR、MNAR
1.对于Cabin分类型数据，缺失率>75%
考虑到实际情况，Cabin首字母通常表示甲板层（如 A、B、C 对应上层甲板，D、E 对应中层，F、G 为下层），所以Cabin的位置可能影响生存率，比如靠近救生艇的舱位可能更容易逃生。而不同等级的舱位（Pclass）可能分布在不同的甲板位置，所以Pclass和Cabin的位置可能存在相关性。
我们用类似假设检验的方法，对缺失值类型进行判断：
step1:业务性判断MCAR与MAR的取舍。检查Cabin字段的缺失情况与Pclass的关系，发现Pclass 与 Cabin 缺失及甲板层（Deck）显著相关→排除MCAR可能性，粗略（统计分析少）判断Cabin缺失值类型为MAR。
step2:统计性判断MNAR的可能性。通过卡方检验（分类变量关联性），p_value<0.05，判断出Cabin缺失与Survived关联性强→暗示MNAR的可能性。
所以在此选择将 Deck 作为衍生特征加入模型并创建Has_Cabin标记，删除原始 Cabin 字段。再进行Deck的缺失值填补
对Deck进行缺失值填充：
1.利用Deck与Pclass的关联性，以下为具体做法
法一：把Deck按照Pclass分组，求出每组的众数，对应组的众数进行缺失值填充
法二：把Deck按照Pclass分组，求出每组的Deck的概率分布，对应组的随机选择Deck进行填充
2.基于模型
模型选择：


对Fare、Age连续型数据，缺失率分别为0.2%和20.6%，前者我们选择删除，后者
（二）异常值检测
（三）数据一致性处理
