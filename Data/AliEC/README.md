# AliEC

This dataset is provided by Alimama
Original link: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56#1

| Dataset           | \#User    | \#Item    | \#Inteaction | Sparsity | Interaction Type           | TimeStamp | User Context | Item Context | Interaction Context |
|------------------|-----------|-----------|--------------|----------|----------------------------|-----------|--------------|--------------|---------------------|
| AliEC                                   | 491,647    | 240,130     | 1,366,056   | 99\.9988% | Click | √ | √ | √ |  |

## Introduction

`Ali_Display_Ad_Click` is a dataset of click rate prediction about display Ad, which is displayed on the website of Taobao. The dataset is offered by the company of Alibaba. 

## Data Sets

Table	Description	Feature
Table               Description               Feature
raw_sample          raw training samples	    User ID, Ad ID, nonclk, clk, timestamp
ad_feature          Ad’s basic information	  Ad ID, campaign ID, Cate ID, Brand
user_profile        user profile              User ID, age, gender, etc
raw_behavior_log    User behavior log         User ID, btag, cate, brand, timestamp

### raw_sample

We randomly sampled 1140000 users from the website of Taobao for 8 days of ad display / click logs (26 million records) to form the original sample skeleton. Field description is as follows:
(1) user: User ID(int);
(2) time_stamp: time stamp(Bigint, 1494032110 stands for 2017-05-06 08:55:10);
(3) adgroup_id: adgroup ID(int);
(4) pid: scenario;
(5) noclk: 1 for not click, 0 for click;
(6) clk: 1 for click, 0 for not click;

We used 7 days’s samples as training samples (20170506-20170512), and the last day’s samples as test samples (20170513).

### ad_feature

This data set covers the basic information of all ads in raw_sample. Field description is as follows:
(1) adgroup_id：Ad ID(int) ;
(2) cate_id：category ID;
(3) campaign_id：campaign ID;
(4) brand：brand ID;
(5) customer_id: Advertiser ID;
(6) price: the price of item

One of the ad ID corresponds to an item, an item belongs to a category, an item belongs to a brand.

### user_profile

This data set covers the basic information of 1060000 users in raw_sample.. Field description is as follows: 
(1) userid: user ID;
(2) cms_segid: Micro group ID;
(3) cms_group_id: cms_group_id;
(4) final_gender_code: gender 1 for male , 2 for female
(5) age_level: age_level
(6) pvalue_level: Consumption grade, 1: low,  2: mid,  3: high
(7) shopping_level: Shopping depth, 1: shallow user, 2: moderate user, 3: depth user
(8) occupation: Is the college student 1: yes, 0: no?
(9) new_user_class_level: City level
(10) behavior_log

### raw_behavior_log

This data set covers the shopping behavior in 22 days of all users in raw_sample(totally seven hundred million records). Field description is as follows:
(1) nick: User ID(int);
(2) time_stamp: time stamp(Bigint, 1494032110 stands for 2017-05-06 08:55:10)；
(3) btag: Types of behavior, include the following four:
type      explanation
ipv       browse
cart      add to the shopping cart
fav	      favor
buy	      buy
(4) cate: category ID(int);
(5) brand: brand ID(int);

Here if we use userID and timestamp as primary key, we will find a lot of duplicate records. This is because the behavior of different types of the data are collected from different departments and when packaged together, there are small deviations (i.e. the same two timestamps may be two different time with a relatively small difference).

Typical research topics
Predict the probability of clicking on an ad when impressed based on user’s history shopping behavior.

### Baseline

AUC:0.622

## Reference and Related Publications

1. Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.
2. Guorui Zhou, Chengru Song, Xiaoqiang Zhu, et al. Deep Interest Network for Click-Through Rate Prediction.https://arxiv.org/abs/1706.06978.