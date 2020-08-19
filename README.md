# seq2seq 구현

> 이 repo에서는 **Sequence to Sequence Learning with Neural Networks**논문 및 tensorflow 사이트를 기반으로 seq2seq 한영변환을 구현해보고 BLEU로 성능을 테스트 해보고, 논문리뷰 및 코드 정리를 하였습니다. korpus는 aihub의 한국어-영어 번역 말뭉치를 사용하였습니다.



### 목차

[논문리뷰](#논문리뷰)

[구현과정](#구현과정)

[결과 및 결론](#{결과 및 결론})

---


# 논문리뷰

![1.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/1.PNG?raw=true)

![2.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/2.PNG?raw=true)

![3.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/3.PNG?raw=true)

![4.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/4.PNG?raw=true)

![5.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/5.PNG?raw=true)

![6.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/6.PNG?raw=true)



# 구현과정

#### 1. data 불러와서 train, val, test set으로 분리

```python
import pandas as pd
news_df = pd.read_excel('/gdrive/My Drive/B반/data/kor.xlsx', sheet_name='Sheet1')

train_df, val_df, test_df = news_df.iloc[:50000, 1:],news_df.iloc[50000:63000, 1:], news_df.iloc[63000:, 1:]
```

```python
train_en, test_en, val_en, train_kor, test_kor, val_kor = train_df['en'], test_df['en'], val_df['en'], train_df['ko'], test_df['ko'], val_df['ko']
```



#### 2. preprocessing

- 영문에는 문장 앞뒤로 start, end 태그 추가, 띄어쓰기 기준으로 split
- 한글에는 앞에 end, 뒤에 start 태그 추가하여 역순으로 정렬

```python
def preprocess_sentences_eng(sentences):
    sentence = []
    for line in sentences:
        w = '<start> '+line+' <end>'
        sentence.append(w)
    return sentence
def preprocess_sentences_kor(sentences):
    okt = Okt()
    sentence = []
    for line in sentences:
        tokens = [i[0] for i in okt.pos(line)]
        tokens.insert(0,'<end>')
        tokens.append('<start>')
        sentence.append(' '.join(tokens[::-1]))
    return sentence
```

#### 2-1. 영문을 띄어쓰기 기준으로 토큰화 했을 시 문제점 발견

![image (1)](seq2seq 코드정리.assets/image (1).png)

이와 같이 단어에 . 이붙은 토큰이 생긴다

-> nltk word_tokenize 이용하여 전처리함

```python
def preprocess_sentences_eng(sentences):
    sentence = []
    for line in sentences:
        tokens = word_tokenize(line)
        tokens.insert(0,'<start>')
        tokens.append('<end>')
        sentence.append(' '.join(tokens))
    return sentence
```



#### 3. tokenize - train_set을 이용하여 영문과 한글을 fit 한다

```python
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='oov')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer
```

```python
train_en = preprocess_sentences_eng(train_en)
train_kor = preprocess_sentences_kor(train_kor)
```

- test_set tokenize를 위한 함수

```python
def tokenize_test(sent,lang='en'):
    if lang=='ko':
        tensor = inp_lang.texts_to_sequences(sent)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,maxlen = max_length_inp,
                                                         padding='post')
        return tensor
    else :
        tensor = targ_lang.texts_to_sequences(sent)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,maxlen =max_length_targ,
                                                         padding='post')
        return tensor
```



#### 4. 데이터 셋 생성

```python
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
steps_per_epoch_val = len(input_tensor_val)//BATCH_SIZE #val_loss계산에 필요
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
```



#### 5. 인코더 및 디코더 모델 작성

- Encoder -> GRU 이용/ hidden_state : -0.08~0.08의 값으로 초기화

```python
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.random.uniform(shape=(self.batch_sz, self.enc_units), minval=-0.08, maxval=0.08)
```

```python
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
```



- BahdanauAttention

```python
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```

```python
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
```



- Decoder

```python
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
```

```python
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)
```



#### 6. optimizer : Adam사용, loss function 정의

```python
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
```



#### 7.  저장경로 지정

```python
checkpoint_dir = '/gdrive/My Drive/강의자료/seq2seq_okt'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
```



#### 8. 훈련

- Encoder output과 Encoder hidden state를 반환하는 Encoder 를 통해 입력 전달
- Output 및 hidden state, Decoder Input(<start> 토큰)이 디코더로 전달됨
- Decoder는 prediction과 decoder hidden state 를 반환하고 모델로 전달된 후 loss 계산
- Teacher Forcing 을 통해 디코더에 대한 다음 입력 결정
- Gradient를 계산하고 optimizer 및 back propagation 적용

```python
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

def test_step(inp, targ, enc_hidden):
  loss = 0
 
  enc_out, enc_hidden = encoder(inp, enc_hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

  for t in range(1, targ.shape[1]):
    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
    loss += loss_function(targ[:, t], predictions)
    dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
```

```python
EPOCHS = 10
loss_data = {'loss':[],'val_loss':[]} #시각화를 위한 epoch 별 loss값 저장
for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  val_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  enc_hidden = encoder.initialize_hidden_state()
  for inp, targ in val_dataset.take(steps_per_epoch_val):
    batch_val_loss = test_step(inp, targ, enc_hidden)
    val_loss += batch_val_loss
  

  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)
  loss_data['loss'].append(total_loss / steps_per_epoch)
  loss_data['val_loss'].append(val_loss/steps_per_epoch_val)
  print('Epoch {} Loss {:.4f} val_los {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch, val_loss/steps_per_epoch_val))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```



#### 9. Evaluate 함수

```python
def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentences_kor(sentence)[0]
  inputs = inp_lang.texts_to_sequences([sentence])[0]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot
    result += targ_lang.index_word[predicted_id] + ' '



    # the predicted ID is fed back into the mode

    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
```



#### 10. BLEU 스코어 계산

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
score_list = []
smoothie = SmoothingFunction().method4
for i in range(len(test_kor.values)):
    
    reference = [test_en.values[i].split()]
    result,_,_= evaluate([test_kor.values[i]])
    candidate = result.capitalize().split()
    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    score_list.append(score)
    if i<20:
        print('실제값 : {}\n예측값 : {}'.format(reference, candidate))

score = sum(score_list)/len(score_list)
print('score는 {}'.format(score))
```





# 결과 및 결론

> default : mecab/ rand,uniform initail/ 역방향/ epoch:10

#### 1. initial 방법 **zero** vs **random,uniform[-0.08,0.08]**

   encoder의 은닉층 초기화 방법을 두가지로 나누어 진행


 ```python
   def initialize_hidden_state(self):
       return tf.zeros((self.batch_sz, self.enc_units))
 ```

 ```python
   def initialize_hidden_state(self):
       return tf.random.uniform(shape=(self.batch_sz, self.enc_units), minval=-0.08, maxval=0.08)
 ```



| initial방법 | zero   | rand,uniform |
| ----------- | ------ | ------------ |
| score       | 0.1872 | 0.2012       |



#### 2. Okt vs Mecab + (Okt 전처리)

- Mecab에비해 Okt의 maxlen이 더 짧은데 vocab size는 더 큼
- Mecab 토큰화 방식이 좋은 성능을 보임

|                 | Okt(전처리) | Okt    | Mecab  |
| ----------- | ---------- | ---------- | ---------- |
| score           | 0.1921 | 0.1856 | 0.2012 |
| vocab size      | 28341 | 28361  | 17827  |
| maxlen | 26   | 26     | 32     |



#### 3. input 순서 정방향 vs 역방향



| input 순서 | 정방향 | 역방향 |
| ---------- | ------ | ------ |
| score      | 0.1954 | 0.2012 |




#### 4. 영문 nltk word_tokenize 이용 

|       | 정방향_nltk_ | 역방향_nltk_ |
| ----- | ------------ | ------------ |
| score | 0.2299       | 0.2326       |

|            | 영문 nltk 사용 | 영문 split사용 |
| ---------- | -------------- | -------------- |
| vocab_size | 15640          | 25457          |
| maxlen     | 21             | 19             |

![graph.PNG](https://github.com/Changyoon-Lee/realization_seq2seq/blob/master/image/graph.PNG?raw=true)

---

### 결론

1. **Initializing** 방식

   두가지 방법(zeros / random uniform)중 **random uniform**이 더 높음

2. input 방식은 **정방향에** **비해 역방향의 성능이 좋음**
3. 한글 토큰화 방식 **Mecab****이** **Okt****보다 성능이 좋음**
   - Okt 내에서 전처리 시 성능 향상

4. 영어 토큰화 방식 띄어쓰기 기준으로 한것 보다 **nltk tokenizer 사용**시 성능향상

