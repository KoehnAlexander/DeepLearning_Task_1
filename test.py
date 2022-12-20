import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import matplotlib.pyplot as plt


ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})

tf.random.set_seed(53)
shuffled = ratings.shuffle(100_000, seed=53, reshuffle_each_iteration=True)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))

unique_user_ids = np.unique(np.concatenate(list(user_ids)))


class RankingModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_id = features["user_id"]
        movie_title = features["movie_title"]

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("user_rating")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)

    def couple_movie_recommendation(self, user_id_1, user_id_2, movie_list):
        test_ratings_1 = {}
        test_ratings_2 = {}
        test_ratings_combined = {}
        for movie_title in movie_list:
            test_ratings_1[movie_title] = self({
                "user_id": user_id_1,
                "movie_title": np.array([movie_title])
            })
            test_ratings_2[movie_title] = self({
                "user_id": user_id_2,
                "movie_title": np.array([movie_title])
            })
            test_ratings_combined[movie_title] = (test_ratings_1[movie_title] + test_ratings_2[movie_title]) / 2

        test_ratings_combined = sorted(test_ratings_combined.items(), key=lambda x: x[1], reverse=True)
        print("Recommendation for the two users: \n")
        for title, score in test_ratings_combined[:10]:
            print(f"{title}: {score}")



model = RankingModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(65536).cache()
cached_test = test.batch(16384).cache()

history = model.fit(cached_train, epochs=50, validation_data=cached_test)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.title("model error")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


evaluation = model.evaluate(cached_test, return_dict=True)


user_1 = np.array(["42"])
user_2 = np.array(["72"])
test_movie_titles, ind = np.unique(np.concatenate(list(movie_titles)), return_index=True)
test = test_movie_titles[np.argsort(ind)]

model.couple_movie_recommendation(user_1, user_2, test[:100])