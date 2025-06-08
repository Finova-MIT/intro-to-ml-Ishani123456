{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8ea199c4-a537-4d2c-a5f5-bd3d7308756d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Load data\n",
    "X, y = load_iris(return_X_y=True)\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "# Train classifier\n",
    "clf = DecisionTreeClassifier(criterion='gini', max_depth=3)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Evaluate\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b728fdf8-985f-4310-96ba-e6cedd7dec47",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MSE: 0.5210801561811792\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# Load data\n",
    "X, y = fetch_california_housing(return_X_y=True)\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "# Train regressor\n",
    "reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5)\n",
    "reg.fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred = reg.predict(X_test)\n",
    "\n",
    "# Evaluate\n",
    "print(\"MSE:\", mean_squared_error(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1f3df2e7-9a78-458f-8c3d-a23525865ebf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (PCA/oblique split): 0.9555555555555556\n"
     ]
    }
   ],
   "source": [
    "# Pseudo-implementation - real implementation is complex\n",
    "# Let's simulate a linear split by applying PCA before using a decision tree\n",
    "\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Load and reduce dimensionality\n",
    "X, y = load_iris(return_X_y=True)\n",
    "X_reduced = PCA(n_components=2).fit_transform(X)\n",
    "\n",
    "# Train with reduced features\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)\n",
    "\n",
    "clf = DecisionTreeClassifier(max_depth=3)\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "print(\"Accuracy (PCA/oblique split):\", accuracy_score(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "014b9bd5-94b1-4ad6-ab7d-b7e0bc719bde",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
