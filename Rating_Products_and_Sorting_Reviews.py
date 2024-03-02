import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.max_rows",None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.describe().T
df.info()
############# Calculate the Average Rating based on current comments and compare it with the existing average rating. #############

df.groupby("asin").agg({"overall":"mean"})

def time_weight_sort(w1,w2,w3):
    return df.loc[df["day_diff"] <=281,"overall"].mean() *w1/100 +\
    df.loc[(df["day_diff"] >281) & (df["day_diff"] <=431),"overall" ].mean() *w2/100 +\
    df.loc[(df["day_diff"] >431) & (df["day_diff"] <=601),"overall" ].mean() *w3/100 +\
    df.loc[(df["day_diff"] >601) ,"overall" ].mean() *w3/100


df["time_based_overall"] = time_weight_sort(39,37,24)

df.sort_values("overall",ascending=False).head(20)
df.sort_values("time_based_overall",ascending=False).head(20)



############# Determine 20 reviews to be displayed on the product detail page for the product. #############

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName","overall","summary","helpful_yes","helpful_no","total_vote","reviewTime"]]

df.head(20)

def score_pos_neg_diff(up,down):
    return up-down

def score_average_rating(up,down):
    if up + down ==0:
        return 0
    return up / (up+down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 is marked as negative and 4-5 is marked as positive and can be adapted to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x:score_pos_neg_diff(x["helpful_yes"],x["helpful_no"]),axis=1)

df.sort_values("score_pos_neg_diff",ascending=False).head(20)

df["score_average_rating"] = df.apply(lambda x:score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)

df.sort_values("score_average_rating",ascending=False).head(20)

df["wilson_lower_bound"] = df.apply(lambda x:wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

df.sort_values("wilson_lower_bound",ascending=False).head(20)

















