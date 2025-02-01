import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import warnings
import datetime

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def gaussian_interaction_profile_kernel(matrix, gamma=1):
    """Compute Gaussian Interaction Profile kernel."""
    sq_dists = np.square(matrix[:, np.newaxis] - matrix).sum(axis=2)
    kernel = np.exp(-gamma * sq_dists)
    return kernel

def prepare_data(separate=False):
    print("loading data")
    
    disease_fea_df = pd.read_csv('drug_str_sim.csv', encoding='ISO-8859-1')
    circRNA_fea_df = pd.read_csv('gene_seq_sim.csv', encoding='ISO-8859-1')
    interaction_df = pd.read_csv('association.csv', encoding='ISO-8859-1')
    
    interaction_df = interaction_df.astype(int)
    
    if interaction_df.shape[0] > circRNA_fea_df.shape[0]:
        interaction_df = interaction_df.iloc[:circRNA_fea_df.shape[0], :]
    if interaction_df.shape[1] > disease_fea_df.shape[0]:
        interaction_df = interaction_df.iloc[:, :disease_fea_df.shape[0]]

    interaction_df = interaction_df.astype(int)
    
    disease_fea_df.to_csv('drug_str_sim.txt', sep='\t', header=False, index=False, float_format='%.0f')
    circRNA_fea_df.to_csv('gene_seq_sim.txt', sep='\t', header=False, index=False, float_format='%.0f')
    interaction_df.to_csv('association.txt', sep='\t', header=False, index=False)
    print("转换完成：CSV 文件已保存为 TXT 文件。")
    
    disease_fea = np.loadtxt("drug_str_sim.txt", dtype=float, delimiter="\t")
    circRNA_fea = np.loadtxt("gene_seq_sim.txt", dtype=float, delimiter="\t")
    interaction = np.loadtxt("association.txt", dtype=int, delimiter="\t")
    
    print(f"disease_fea shape: {disease_fea.shape}")
    print(f"circRNA_fea shape: {circRNA_fea.shape}")
    
    if interaction.shape[0] > circRNA_fea.shape[0] or interaction.shape[1] > disease_fea.shape[0]:
        raise ValueError("Interaction file dimensions are larger than feature file dimensions.")
    
    # 计算 GIP 相似性
    circRNA_gip = gaussian_interaction_profile_kernel(interaction)
    drug_gip = gaussian_interaction_profile_kernel(interaction.T)
    
    link_number = 0
    train_circRNA = []  
    train_drug = []
    test_circRNA = []       
    test_drug = []
    label1 = []
    label2 = []
    label22 = []
    ttfnl_circRNA = []
    ttfnl_drug = []
    
    for i in range(interaction.shape[0]):
        for j in range(interaction.shape[1]):
            if i >= circRNA_fea.shape[0] or j >= disease_fea.shape[0]:
                continue
            
            if interaction[i, j] == 1:
                label1.append(interaction[i, j])
                link_number += 1
                circRNA_fea_tmp = np.hstack((circRNA_fea[i], circRNA_gip[i]))  # 融合特征
                disease_fea_tmp = np.hstack((disease_fea[j], drug_gip[j]))  # 融合特征
                train_circRNA.append(circRNA_fea_tmp)
                train_drug.append(disease_fea_tmp)
            elif interaction[i, j] == 0:
                label2.append(interaction[i, j])
                circRNA_fea_tmp1 = np.hstack((circRNA_fea[i], circRNA_gip[i]))  # 融合特征
                disease_fea_tmp1 = np.hstack((disease_fea[j], drug_gip[j]))  # 融合特征
                test_circRNA.append(circRNA_fea_tmp1)
                test_drug.append(disease_fea_tmp1)
    
    print("link_number", link_number)
    print("no_link_number", len(label2))
    m = np.arange(len(test_circRNA))
    np.random.shuffle(m)
    
    for x in m[:5000]:
        ttfnl_circRNA.append(test_circRNA[x])
        ttfnl_drug.append(test_drug[x])
        label22.append(label2[x])
             
    for x in range(100):
        tfnl_circRNA = ttfnl_circRNA[x]
        tfnl_drug = ttfnl_drug[x]
        lab = label22[x]
        train_circRNA.append(tfnl_circRNA)
        train_drug.append(tfnl_drug)
        label1.append(lab)
    
    return (np.array(train_circRNA, dtype=np.float32), 
            np.array(train_drug, dtype=np.float32), 
            np.array(label1), 
            np.array(ttfnl_circRNA, dtype=np.float32), 
            np.array(ttfnl_drug, dtype=np.float32))

def multi_scale_conv_network(input_dim):
    input_layer = Input(shape=(input_dim, 1))  # 增加了通道维度
    
    conv_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.01))(input_layer)
    conv_2 = Conv1D(filters=64, kernel_size=5, padding='same', activation=LeakyReLU(alpha=0.01))(input_layer)
    conv_3 = Conv1D(filters=64, kernel_size=7, padding='same', activation=LeakyReLU(alpha=0.01))(input_layer)
    
    concatenated = Concatenate(axis=-1)([conv_1, conv_2, conv_3])
    
    flattened = Flatten()(concatenated)
    dense_1 = Dense(512, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(flattened)
    dense_1 = Dropout(0.4)(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    
    return Model(inputs=input_layer, outputs=dense_1)

train_circRNA, train_drug, label1, test_circRNA, test_drug = prepare_data(separate=True)
print("Data preparation complete.")

# 构建多尺度卷积神经网络
circRNA_input_dim = train_circRNA.shape[1]
drug_input_dim = train_drug.shape[1]

circRNA_fusion_net = multi_scale_conv_network(circRNA_input_dim)
drug_fusion_net = multi_scale_conv_network(drug_input_dim)

# 处理特征并得到融合后的新特征
fused_train_circRNA = circRNA_fusion_net.predict(train_circRNA[..., np.newaxis])  # 添加通道维度
fused_train_drug = drug_fusion_net.predict(train_drug[..., np.newaxis])  # 添加通道维度
fused_test_circRNA = circRNA_fusion_net.predict(test_circRNA[..., np.newaxis])  # 添加通道维度
fused_test_drug = drug_fusion_net.predict(test_drug[..., np.newaxis])  # 添加通道维度

# 将处理后的特征进行拼接，作为输入
train_features = np.concatenate((fused_train_circRNA, fused_train_drug), axis=1)
test_features = np.concatenate((fused_test_circRNA, fused_test_drug), axis=1)

# 使用 SMOTE 平衡数据集
def balance_data_with_smote(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

train_features_balanced, train_labels_balanced = balance_data_with_smote(train_features, label1)

# 调整数据集的平衡性
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels_balanced), y=train_labels_balanced)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1
                
    acc = float(tp + tn) / test_num
    
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)

    return acc, precision, sensitivity, specificity, MCC, f1_score

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = tf.keras.utils.to_categorical(y)
    return y, encoder

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (tf.keras.backend.ones_like(y_pred) - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_pred) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

def DNN():
    model = Sequential()
    model.add(Dense(units=512, input_shape=(512,), activation=LeakyReLU(alpha=0.01), kernel_initializer='glorot_normal', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(units=512, activation=LeakyReLU(alpha=0.01), kernel_initializer='glorot_normal', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(units=256, activation=LeakyReLU(alpha=0.01), kernel_initializer='glorot_normal', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(units=2, activation='sigmoid', kernel_initializer='glorot_normal'))
    rmsprop = RMSprop(learning_rate=0.0001)  # 调整优化器为 RMSprop 并微调学习率
    model.compile(loss='binary_crossentropy', optimizer=rmsprop)
    return model

def AAE(x_train):
    encoding_dim = 512  # 将编码维度修改为512
    input_dim = x_train.shape[1]
    input_img = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(350, activation=LeakyReLU(alpha=0.01))(input_img)
    encoded = Dense(250, activation=LeakyReLU(alpha=0.01))(encoded)
    encoded = Dense(100, activation=LeakyReLU(alpha=0.01))(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    
    # Decoder
    decoded = Dense(100, activation=LeakyReLU(alpha=0.01))(encoder_output)
    decoded = Dense(250, activation=LeakyReLU(alpha=0.01))(decoded)
    decoded = Dense(350, activation=LeakyReLU(alpha=0.01))(decoded)
    decoded = Dense(input_dim, activation='tanh')(decoded)
    
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    
    # Discriminator
    discr_input = Input(shape=(encoding_dim,))
    discr = Dense(100, activation=LeakyReLU(alpha=0.01))(discr_input)
    discr = Dense(1, activation='sigmoid')(discr)
    discriminator = Model(inputs=discr_input, outputs=discr)
    
    discriminator.compile(optimizer=Adam(), loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Training autoencoder
    autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, shuffle=True)
    
    # Adversarial training
    for epoch in range(30):
        # Select a random batch of images
        idx = np.random.randint(0, x_train.shape[0], 128)
        imgs = x_train[idx]
        
        # Encode images
        encoded_imgs = encoder.predict(imgs)
        
        # Generate fake samples
        fake = np.random.normal(size=(128, encoding_dim))
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(encoded_imgs, np.ones((128, 1)))
        d_loss_fake = discriminator.train_on_batch(fake, np.zeros((128, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator (encoder)
        g_loss = autoencoder.train_on_batch(imgs, imgs)
        
        print(f'{epoch} [D loss: {d_loss}] [G loss: {g_loss}]')
    
    encoded_imgs = encoder.predict(x_train)
    return encoder, autoencoder, encoded_imgs


def DeepCDA():
    # 初始化数据集
    train_circRNA, train_drug, label1, test_circRNA, test_drug = prepare_data(separate=True)
    print("Data preparation complete.")

    circRNA_input_dim = train_circRNA.shape[1]
    drug_input_dim = train_drug.shape[1]

    # 构建多尺度卷积神经网络
    circRNA_fusion_net = multi_scale_conv_network(circRNA_input_dim)
    drug_fusion_net = multi_scale_conv_network(drug_input_dim)

    # 处理特征并得到融合后的新特征
    fused_train_circRNA = circRNA_fusion_net.predict(train_circRNA[..., np.newaxis])
    fused_train_drug = drug_fusion_net.predict(train_drug[..., np.newaxis])
    fused_test_circRNA = circRNA_fusion_net.predict(test_circRNA[..., np.newaxis])
    fused_test_drug = drug_fusion_net.predict(test_drug[..., np.newaxis])

    # 将处理后的特征进行拼接，作为输入
    train_features = np.concatenate((fused_train_circRNA, fused_train_drug), axis=1)
    test_features = np.concatenate((fused_test_circRNA, fused_test_drug), axis=1)

    # 使用 SMOTE 平衡数据集
    train_features_balanced, train_labels_balanced = balance_data_with_smote(train_features, label1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels_balanced), y=train_labels_balanced)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    X_data1 = train_features_balanced
    labels = train_labels_balanced
    
    print(f"X_data1 shape: {X_data1.shape}")
    
    y, encoder = preprocess_labels(labels)
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num].astype(np.float32)
    y = y[num]
    
    encoder, autoencoder, encoded_imgs = AAE(X_data1)

    num_cross_val = 5  # 可以更改为10做10折交叉验证
    all_performance_DNN = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(14, 6))
    
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(encoded_imgs) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(encoded_imgs) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])

        real_labels = [0 if val[0] == 1 else 1 for val in test_label]
        train_label_new = [0 if val[0] == 1 else 1 for val in train_label]

        model_DNN = DNN()
        train_label_new_forDNN = np.array([[0, 1] if i == 1 else [1, 0] for i in train_label_new])
        
        # 加入 class_weight 参数
        model_DNN.fit(train1, train_label_new_forDNN, batch_size=64, epochs=30, shuffle=True, class_weight=class_weights)
        
        proba = model_DNN.predict_classes(test1, batch_size=64, verbose=True)
        ae_y_pred_prob = model_DNN.predict_proba(test1, batch_size=64, verbose=True)
        
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), proba, real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:, 1])
        aupr_score = auc(recall, precision1)

        print(f"AUTO-DNN: {acc}, {precision}, {sensitivity}, {specificity}, {MCC}, {auc_score}, {aupr_score}, {f1_score}")

        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])
        
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        mean_precision += np.interp(mean_recall, recall[::-1], precision1[::-1])
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC fold {fold+1} (AUC = {auc_score:.4f})')
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision1, label=f'PR fold {fold+1} (AP = {aupr_score:.4f})')
    
    mean_tpr /= num_cross_val
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    mean_precision /= num_cross_val
    mean_ap = auc(mean_recall, mean_precision)
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (AUC = {mean_auc:.4f})', lw=2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characteristic curve: 5-Fold CV')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(mean_recall, mean_precision, 'k--', label=f'Mean PR (AP = {mean_ap:.4f})', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: 5-Fold CV')
    plt.legend(loc='lower left')
    
    plt.tight_layout()

    # 使用时间戳生成文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'roc_pr_curves_{timestamp}.png')  # 保存图像为新的文件
    plt.show()

    mean_performance = np.mean(np.array(all_performance_DNN), axis=0)
    print(f'Mean performance of AE-DNN:')
    print(f'Mean-Accuracy={mean_performance[0]}, Mean-precision={mean_performance[1]}')
    print(f'Mean-Sensitivity={mean_performance[2]}, Mean-Specificity={mean_performance[3]}')
    print(f'Mean-MCC={mean_performance[4]}, Mean-AUC={mean_performance[5]}')
    print(f'Mean-AUPR={mean_performance[6]}, Mean-F1={mean_performance[7]}')

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label

if __name__ == "__main__":
    DeepCDA()
