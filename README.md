# KaggleLearning
Personal Notes

## LMSYS - Chatbot Arena Human Preference Predictions
预测在两个大型语言模型（LLMs）驱动的聊天机器人之间的正面交锋中，用户最可能偏好哪种回答，确保LLM生成的回应能够与用户的偏好相符合。

###  https://www.kaggle.com/code/awsaf49/lmsys-kerasnlp-starter

很基础很详细的一些解说，顺着跑了一遍。

1. **环境设置**（安装必要的库，选择合适的后端）

2. **数据理解和预处理**（加载，理解数据集结构，预处理数据）

3. **模型配置**（定义模型参数，数据分割，预处理设置）

4. **数据加载器的建立**（利用tf.data.Dataset建立训练和验证数据加载器）

5. **模型建立，训练和性能监测**（构建模型，编译模型，训练模型，并使用回调函数进行模型性能监测和保存最佳模型）

6. **模型预测和提交文件的准备**（加载最佳模型进行预测，准备提交到竞赛平台的文件）

### https://www.kaggle.com/code/siddhvr/lmsys-cahpp-llama3-8b-inference-baseline

使用LLM（大型语言模型）Llama 3和机器学习算法（如LightGBM）进行模型融合（不同的模型可能从数据中学习到不同的特征和模式，通过合理地结合这些模型的预测，可以获得比任何单个模型都更好的结果）,预测聊天机器人对话结果。具体实现就是应用两种不同的方法，再把这两种方法的预测结果进行加权融合。

1. **环境配置与库导入：** 包括`transformers`, `tokenizers`, `bitsandbytes`, `peft`之类，使用预训练的语言模型和进行文本处理。

2. **数据预处理：** 将字符串列表转换为单个字符串并准备模型所需的格式。 

3. **模型加载与配置：** 使用`transformers`库从一个预训练的Llama模型加载了一个用于序列分类的模型配置，并使用`BitsAndBytesConfig`对模型进行了性能优化的配置。为两个不同的GPU配置了两个模型实例，并且加载了预先训练好的权重。

4. **特征提取与模型预测：** 对于LightGBM部分，用`CountVectorizer`从训练和测试文本中提取特征，然后加载LightGBM模型，对测试集做预测。对于Llama模型，用AutoTokenizer对测试数据做标记化，然后在两个GPU上并行运行模型生成预测。

5. **融合预测结果：** 把来自LightGBM和Llama模型的预测结果进行了加权融合
```python
preds = 0.2 * lgb_preds + 0.8 * llama_preds
```
