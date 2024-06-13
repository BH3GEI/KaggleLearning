# KaggleLearning
Personal Notes

## LMSYS - Chatbot Arena Human Preference Predictions
预测在两个大型语言模型（LLMs）驱动的聊天机器人之间的正面交锋中，用户最可能偏好哪种回答，确保LLM生成的回应能够与用户的偏好相符合。

###  https://www.kaggle.com/code/awsaf49/lmsys-kerasnlp-starter

很基础很详细的一些解说，顺着跑了一遍。

1. **环境设置**（安装必要的库，选择jax后端）

2. **数据预处理**（加载，理解数据集结构，预处理数据）

3. **模型配置**（定义模型参数，数据分割，预处理设置）

4. **数据加载器的建立**（用tf.data.Dataset建立训练和验证数据加载器）

5. **模型建立，训练和性能监测**（构建，编译，训练，并使用回调函数进行模型性能监测和保存最佳模型）


### https://www.kaggle.com/code/siddhvr/lmsys-cahpp-llama3-8b-inference-baseline

使用LLM（大型语言模型）Llama 3和机器学习算法（如LightGBM）进行模型融合（不同的模型可能从数据中学习到不同的特征和模式，通过合理地结合这些模型的预测，可以获得比任何单个模型都更好的结果）,预测聊天机器人对话结果。具体实现就是应用两种不同的方法，再把这两种方法的预测结果进行加权融合。

1. **库导入：** 包括`transformers`, `tokenizers`, `bitsandbytes`, `peft`之类。
```python
!pip install -q -U bitsandbytes --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U transformers --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U tokenizers --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U peft --no-index --find-links ../input/llm-detect-pip/
```
2. **数据预处理：** 执行了一个简单的预处理函数process，将多个字符串合并成单个长字符串，并通过添加特定的前缀（如"User prompt"、"Model A"、"Model B"等）对文本进行格式化。
```python
import pandas as pd

# 加载测试集
test = pd.read_csv('/path/to/your/test.csv')

# 预处理函数
def process(input_str):
    """
    处理单条数据。
    - 去掉字符串列表表示的外层括号。
    - 将字符串列表中的元素拼接成单一字符串。
    """
    if not isinstance(input_str, str):
        return ""
    
    # 去除字符串列表的括号，并根据","分割字符串
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    
    # 将分割后的字符串列表连接成单一字符串
    return ' '.join(sentences)

# 调用函数
test.loc[:, 'prompt'] = test['prompt'].apply(process)
test.loc[:, 'response_a'] = test['response_a'].apply(process)
test.loc[:, 'response_b'] = test['response_b'].apply(process)

# 合出来一个整体的长字符串
test['text'] = 'User prompt: ' + test['prompt'] + '\n\nModel A :\n' + test['response_a'] + '\n\n--------\n\nModel B:\n' + test['response_b']

print(test['text'][0])  
```

3. **模型加载与配置：** 使用`transformers`库从一个预训练的Llama模型加载了一个用于序列分类的模型配置，并使用`BitsAndBytesConfig`对模型进行了性能优化的配置。为两个不同的GPU配置了两个模型实例，并且加载了预先训练好的权重。
```python
# 加载AutoTokenizer进行文本的标记化
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 配置BitsAndBytes以优化性能
bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16, bnb_8bit_use_double_quant=False)

# 根据配置和模型路径加载Llama模型
base_model_0 = LlamaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, torch_dtype=torch.float16, quantization_config=bnb_config, device_map='cuda:0')
base_model_0.config.pad_token_id = tokenizer.pad_token_id
```
4. **特征提取与模型预测：** 对于LightGBM部分，用`CountVectorizer`从训练和测试文本中提取特征，然后加载LightGBM模型，对测试集做预测。对于Llama模型，用AutoTokenizer对测试数据做标记化，然后在两个GPU上并行运行模型生成预测。
```python
tokens = tokenizer(test['text'].tolist(), padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
INPUT_IDS = tokens['input_ids'].to(DEVICE, dtype=torch.int32)
ATTENTION_MASKS = tokens['attention_mask'].to(DEVICE, dtype=torch.int32)
```
5. **融合预测结果：** 把来自LightGBM和Llama模型的预测结果进行了加权融合
```python
preds = 0.2 * lgb_preds + 0.8 * llama_preds
```
