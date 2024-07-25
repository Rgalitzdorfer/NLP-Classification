import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
# Step 1: Load the Dataset
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv'
data = pd.read_csv(file_path)
print(tf.__version__)
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
# Inspect the class distribution
print(data['Label'].value_counts())

# Step 2: Preprocess the Data
# Separate the majority and minority classes
majority_class = data[data['Label'] == 'RF']
minority_classes = data[data['Label'] != 'RF']

# Encode the labels
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Normalize the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop('Label', axis=1))

# Separate features and labels
X = data_scaled
y = data['Label']

# Define the GAN architecture
# Step 3: Build the GAN Model
def build_generator(latent_dim, n_outputs):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_outputs, activation='tanh'))
    return model

def build_discriminator(n_inputs):
    model = Sequential()
    model.add(Dense(512, input_dim=n_inputs))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    discriminator.trainable = False
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

latent_dim = 100
n_features = X.shape[1]

generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator(n_features)
gan = build_gan(generator, discriminator)

# Step 4: Train the GAN Model
def train_gan(gan, generator, discriminator, X_train, epochs=10000, batch_size=64):
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_samples = X_train[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

train_gan(gan, generator, discriminator, X)

# Step 5: Generate Synthetic Data
def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    return synthetic_data

# Generate synthetic data for minority classes
num_samples = len(majority_class) - len(minority_classes)
synthetic_data = generate_synthetic_data(generator, num_samples)
synthetic_data = scaler.inverse_transform(synthetic_data)
synthetic_labels = np.random.choice(minority_classes['Label'].unique(), num_samples)

# Combine the synthetic data with the original dataset
synthetic_df = pd.DataFrame(synthetic_data, columns=data.drop('Label', axis=1).columns)
synthetic_df['Label'] = synthetic_labels

augmented_data = pd.concat([data, synthetic_df])

# Step 6: Save the Generated Data
output_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 9/Augmented_Data.csv'
augmented_data.to_csv(output_path, index=False)
print(f"Synthetic data saved to {output_path}")
