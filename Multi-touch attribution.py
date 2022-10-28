# Databricks notebook source
# MAGIC %md
# MAGIC ## 1: Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1: View dataset

# COMMAND ----------

display(spark.read.format('csv').option('header','true').load('/FileStore/tables/attribution_data.csv').limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2: Import libraries

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import *
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3: Define schema and read the data into a dataframe

# COMMAND ----------

schema = StructType([StructField('uid', StringType(), True),
StructField('time', TimestampType(), True),
StructField('interaction', StringType(), True),
StructField('channel', StringType(), True),
StructField('conversion', IntegerType(), True)])

# COMMAND ----------

raw_data_df = spark.read.format("csv") \
            .option("header", "true") \
            .schema(schema) \
            .load('/FileStore/tables/attribution_data.csv')

# COMMAND ----------

display(raw_data_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.4: Write Data to Delta Lake

# COMMAND ----------

bronze_tbl_path = '/FileStore/tables/bronzetable'

# COMMAND ----------

raw_data_df.write.format("delta") \
  .option("checkpointLocation", bronze_tbl_path+"/checkpoint") \
  .save(bronze_tbl_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.5: Create Database

# COMMAND ----------

database_name = "MultitouchAttribution"
# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.6: Create bronze-level table in Delta format

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.bronze
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.7: View the bronze table

# COMMAND ----------

bronze_tbl = spark.table("{}.bronze".format(database_name))

# COMMAND ----------

display(bronze_tbl.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.8: Set the current database so that it doesn't need to be manually specified each time it's used.

# COMMAND ----------

_ = spark.sql("use {}".format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.9: Create a user journey temporary view

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW user_journey_view AS
# MAGIC SELECT
# MAGIC   sub2.uid AS uid,CASE
# MAGIC     WHEN sub2.conversion == 1 then concat('Start > ', sub2.path, ' > Conversion')
# MAGIC     ELSE concat('Start > ', sub2.path, ' > Null')
# MAGIC   END AS path,
# MAGIC   sub2.first_interaction AS first_interaction,
# MAGIC   sub2.last_interaction AS last_interaction,
# MAGIC   sub2.conversion AS conversion,
# MAGIC   sub2.visiting_order AS visiting_order
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       sub.uid AS uid,
# MAGIC       concat_ws(' > ', collect_list(sub.channel)) AS path,
# MAGIC       element_at(collect_list(sub.channel), 1) AS first_interaction,
# MAGIC       element_at(collect_list(sub.channel), -1) AS last_interaction,
# MAGIC       element_at(collect_list(sub.conversion), -1) AS conversion,
# MAGIC       collect_list(sub.visit_order) AS visiting_order
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           uid,
# MAGIC           channel,
# MAGIC           time,
# MAGIC           conversion,
# MAGIC           dense_rank() OVER (
# MAGIC             PARTITION BY uid
# MAGIC             ORDER BY
# MAGIC               time asc
# MAGIC           ) as visit_order
# MAGIC         FROM
# MAGIC           bronze
# MAGIC       ) AS sub
# MAGIC     GROUP BY
# MAGIC       sub.uid
# MAGIC   ) AS sub2;

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.10: View the user journey data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM user_journey_view
# MAGIC limit(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.11: Create and view the gold_user_journey table

# COMMAND ----------

gold_user_journey_tbl_path = '/FileStore/tables/golduserjourneytable'

# COMMAND ----------

_ = spark.sql('''
  CREATE TABLE IF NOT EXISTS `{}`.gold_user_journey
  USING DELTA 
  LOCATION '{}'
  AS SELECT * from user_journey_view
  '''.format(database_name, gold_user_journey_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gold_user_journey
# MAGIC limit (5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.12: Optimize the gold_user_journey table

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE gold_user_journey ZORDER BY uid

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.13: Create temporary view for first-touch and last-touch attribution metrics

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW attribution_view AS
# MAGIC SELECT
# MAGIC   'first_touch' AS attribution_model,
# MAGIC   first_interaction AS channel,
# MAGIC   round(count(*) / (
# MAGIC      SELECT COUNT(*)
# MAGIC      FROM gold_user_journey
# MAGIC      WHERE conversion = 1),2) AS attribution_percent
# MAGIC FROM gold_user_journey
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY first_interaction
# MAGIC UNION
# MAGIC SELECT
# MAGIC   'last_touch' AS attribution_model,
# MAGIC   last_interaction AS channel,
# MAGIC   round(count(*) /(
# MAGIC       SELECT COUNT(*)
# MAGIC       FROM gold_user_journey
# MAGIC       WHERE conversion = 1),2) AS attribution_percent
# MAGIC FROM gold_user_journey
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY last_interaction

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.14: Use the temporary view to create the gold_attribution table

# COMMAND ----------

gold_attribution_tbl_path = '/FileStore/tables/goldattributiontable'

# COMMAND ----------

_ = spark.sql('''
CREATE TABLE IF NOT EXISTS gold_attribution
USING DELTA
LOCATION '{}'
AS
SELECT * FROM attribution_view'''.format(gold_attribution_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_attribution

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.15: Import libraries

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.16: Use the gold_attribution table to view first touch vs. last touch by channel

# COMMAND ----------

attribution_pd = spark.table('gold_attribution').toPandas()

sns.set(font_scale=1.1)
sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=attribution_pd, kind='bar', aspect=2).set_xticklabels(rotation=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2: Production

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Check 1: Upsert data into the gold_user_journey table

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO gold_user_journey
# MAGIC USING user_journey_view
# MAGIC ON user_journey_view.uid = gold_user_journey.uid
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED
# MAGIC   THEN INSERT *

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Check 2: Propogate updates made to the gold_user_journey table to the gold_attribution table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW attribution_view AS
# MAGIC SELECT 'first_touch' AS attribution_model, first_interaction AS channel, 
# MAGIC         round(count(*)/(SELECT COUNT(*) FROM gold_user_journey WHERE conversion =1), 2)AS attribution_percent 
# MAGIC FROM gold_user_journey 
# MAGIC WHERE conversion =1 
# MAGIC GROUP BY first_interaction
# MAGIC UNION
# MAGIC SELECT 'last_touch' AS attribution_model, last_interaction AS channel, 
# MAGIC         round(count(*)/(SELECT COUNT(*) FROM gold_user_journey WHERE conversion =1), 2)AS attribution_percent 
# MAGIC FROM gold_user_journey 
# MAGIC WHERE conversion =1 
# MAGIC GROUP BY last_interaction

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO gold_attribution
# MAGIC USING attribution_view
# MAGIC ON attribution_view.attribution_model = gold_attribution.attribution_model AND attribution_view.channel = gold_attribution.channel
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED
# MAGIC   THEN INSERT *

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Check 3: Review Delta Lake table history for auditing & governance

# COMMAND ----------

# MAGIC %sql
# MAGIC describe history gold_user_journey

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3: Multi-Touch Attribution with Markov Chains

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1: Import libraries

# COMMAND ----------

from pyspark.sql.types import StringType, ArrayType
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### 3.2: Define a user-defined function (UDF) that takes a customer journey and enumerates each of the corresponding channel transitions

# COMMAND ----------

 def get_transition_array(path):
  '''
    This function takes as input a user journey (string) where each state transition is marked by a >. 
    The output is an array that has an entry for each individual state transition.
  '''
  state_transition_array = path.split(">")
  initial_state = state_transition_array[0]
  
  state_transitions = []
  for state in state_transition_array[1:]:
    state_transitions.append(initial_state.strip()+' > '+state.strip())
    initial_state =  state
  
  return state_transitions

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### 3.3: Register the `get_transition_array` udf as a Spark UDF so that it can be utilized in Spark SQL

# COMMAND ----------

spark.udf.register("get_transition_array", get_transition_array, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### 3.4: Use the `get_transition_array` to enumerate all channel transitions in a customer's journey

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW markov_state_transitions AS
# MAGIC SELECT path,
# MAGIC   explode(get_transition_array(path)) as transition,
# MAGIC   1 AS cnt
# MAGIC FROM
# MAGIC   gold_user_journey

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from markov_state_transitions
# MAGIC limit(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.5: Construct the transition probability matrix

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW transition_matrix AS
# MAGIC SELECT
# MAGIC   left_table.start_state,
# MAGIC   left_table.end_state,
# MAGIC   left_table.total_transitions / right_table.total_state_transitions_initiated_from_start_state AS transition_probability
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       transition,
# MAGIC       sum(cnt) total_transitions,
# MAGIC       trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC       trim(SPLIT(transition, '>') [1]) end_state
# MAGIC     FROM
# MAGIC       markov_state_transitions
# MAGIC     GROUP BY
# MAGIC       transition
# MAGIC     ORDER BY
# MAGIC       transition
# MAGIC   ) left_table
# MAGIC   JOIN (
# MAGIC     SELECT
# MAGIC       a.start_state,
# MAGIC       sum(a.cnt) total_state_transitions_initiated_from_start_state
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC           cnt
# MAGIC         FROM
# MAGIC           markov_state_transitions
# MAGIC       ) AS a
# MAGIC     GROUP BY
# MAGIC       a.start_state
# MAGIC   ) right_table ON left_table.start_state = right_table.start_state
# MAGIC ORDER BY
# MAGIC   end_state DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.6: Validate that the state transition probabilities are calculated correctly

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT start_state, round(sum(transition_probability),2) as transition_probability_sum 
# MAGIC FROM transition_matrix
# MAGIC GROUP BY start_state

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.7: Display the transition probability matrix

# COMMAND ----------

transition_matrix_pd = spark.table('transition_matrix').toPandas()
transition_matrix_pivot = transition_matrix_pd.pivot(index='start_state',columns='end_state',values='transition_probability')

plt.figure(figsize=(10,5))
sns.set(font_scale=1.4)
sns.heatmap(transition_matrix_pivot,cmap='Blues',vmax=0.25,annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.8: Define `get_transition_probability_graph` utility function

# COMMAND ----------

def get_transition_probability_graph(removal_state = "null"):
  '''
  This function calculates a subset of the transition probability graph based on the state to exclude
      removal_state: channel that we want to exclude from our Transition Probability Matrix
  returns subset of the Transition Probability matrix as pandas Dataframe
  '''
  
  transition_probability_pandas_df = None
  
  # Get the transition probability graph without any states excluded if the removal_state is null
  if removal_state == "null":
    transition_probability_pandas_df = spark.sql('''select
        trim(start_state) as start_state,
        collect_list(end_state) as next_stages,
        collect_list(transition_probability) as next_stage_transition_probabilities
      from
        transition_matrix
      group by
        start_state''').toPandas()
    
  # Otherwise, get the transition probability graph with the specified channel excluded/removed
  else:
    transition_probability_pandas_df = spark.sql('''select
      sub1.start_state as start_state,
      collect_list(sub1.end_state) as next_stages,
      collect_list(transition_probability) as next_stage_transition_probabilities
      from
      (
        select
          trim(start_state) as start_state,
          case
            when end_state == \"'''+removal_state+'''\" then 'Null'
            else end_state
          end as end_state,
          transition_probability
        from
          transition_matrix
        where
          start_state != \"'''+removal_state+'''\"
      ) sub1 group by sub1.start_state''').toPandas()

  return transition_probability_pandas_df

# COMMAND ----------

transition_probability_pandas_df = get_transition_probability_graph()

# COMMAND ----------

transition_probability_pandas_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.9: Define `calculate_conversion_probability` utility function

# COMMAND ----------

def calculate_conversion_probability(transition_probability_pandas_df, calculated_state_conversion_probabilities, visited_states, current_state="Start"):
  '''
  This function calculates total conversion probability based on a subset of the transition probability graph
    transition_probability_pandas_df: This is a Dataframe that maps the current state to all probable next stages along with their transition probability
    removal_state: the channel that we want to exclude from our Transition Probability Matrix
    visited_states: set that keeps track of the states that have been visited thus far in our state transition graph.
    current_state: by default the start state for the state transition graph is Start state
  returns conversion probability of current state/channel 
  '''
 
  #If the customer journey ends with conversion return 1
  if current_state=="Conversion":
    return 1.0
  
  #If the customer journey ends without conversion, or if we land on the same state again, return 0.
  #Note: this step will mitigate looping on a state in the event that a customer path contains a transition from a channel to that same channel.
  elif (current_state=="Null") or (current_state in visited_states):
    return 0.0
  
  #Get the conversion probability of the state if its already calculated
  elif current_state in calculated_state_conversion_probabilities.keys():
    return calculated_state_conversion_probabilities[current_state]
  
  else:
  #Calculate the conversion probability of the new current state
    #Add current_state to visited_states
    visited_states.add(current_state)
    
    #Get all of the transition probabilities from the current state to all of the possible next states
    current_state_transition_df = transition_probability_pandas_df.loc[transition_probability_pandas_df.start_state==current_state]
    
    #Get the next states and the corresponding transition probabilities as a list.
    next_states = current_state_transition_df.next_stages.to_list()[0]
    next_states_transition_probab = current_state_transition_df.next_stage_transition_probabilities.to_list()[0]
    
    #This will hold the total conversion probability of each of the states that are candidates to be visited next from the current state.
    current_state_conversion_probability_arr = []
    
    #Call this function recursively until all states in next_states have been incorporated into the total conversion probability
    import copy
    #Loop over the list of next states and their transition probabilities recursively
    for next_state, next_state_tx_probability in zip(next_states, next_states_transition_probab):
      current_state_conversion_probability_arr.append(next_state_tx_probability * calculate_conversion_probability(transition_probability_pandas_df, calculated_state_conversion_probabilities, copy.deepcopy(visited_states), next_state))
    
    #Sum the total conversion probabilities we calculated above to get the conversion probability of the current state.
    #Add the conversion probability of the current state to our calculated_state_conversion_probabilities dictionary.
    calculated_state_conversion_probabilities[current_state] =  sum(current_state_conversion_probability_arr)
    
    #Return the calculated conversion probability of the current state.
    return calculated_state_conversion_probabilities[current_state]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.10: Calculate the total conversion probability 

# COMMAND ----------

total_conversion_probability = calculate_conversion_probability(transition_probability_pandas_df, {}, visited_states=set(), current_state="Start")

# COMMAND ----------

total_conversion_probability

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.11: Calculate the removal effect per channel

# COMMAND ----------

removal_effect_per_channel = {}
for channel in transition_probability_pandas_df.start_state.to_list():
  if channel!="Start":
    transition_probability_subset_pandas_df = get_transition_probability_graph(removal_state=channel)
    new_conversion_probability =  calculate_conversion_probability(transition_probability_subset_pandas_df, {}, visited_states=set(), current_state="Start")
    removal_effect_per_channel[channel] = round(((total_conversion_probability-new_conversion_probability)/total_conversion_probability), 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.12: Calculate conversion attribution per channel

# COMMAND ----------

conversion_attribution={}

for channel in removal_effect_per_channel.keys():
  conversion_attribution[channel] = round(removal_effect_per_channel[channel] / sum(removal_effect_per_channel.values()), 2)

channels = list(conversion_attribution.keys())
conversions = list(conversion_attribution.values())

conversion_pandas_df= pd.DataFrame({'attribution_model': 
                                    ['markov_chain' for _ in range(len(channels))], 
                                    'channel':channels, 
                                    'attribution_percent': conversions})


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.13: Register `conversion_pandas_df` as table to use SQL

# COMMAND ----------

sparkDF=spark.createDataFrame(conversion_pandas_df) 
sparkDF.createOrReplaceTempView("markov_chain_attribution_update")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.14: View channel attribution

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from markov_chain_attribution_update

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.15: Merge channel attribution results into `gold_attribution` table

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO gold_attribution
# MAGIC USING markov_chain_attribution_update
# MAGIC ON markov_chain_attribution_update.attribution_model = gold_attribution.attribution_model AND markov_chain_attribution_update.channel = gold_attribution.channel
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED
# MAGIC   THEN INSERT *

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.16: Compare Channel Performance Across Methods

# COMMAND ----------

attribution_pd = spark.table('gold_attribution').toPandas()

sns.set(font_scale=1.1)
sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=attribution_pd, kind='bar', aspect=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4: Spend Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1: Import libraries

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1.4)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2: Create ad spend table

# COMMAND ----------

gold_ad_spend_tbl_path = '/FileStore/tables/goldadspendtable'

# COMMAND ----------

_ = spark.sql('''
  CREATE OR REPLACE TABLE gold_ad_spend (
    campaign_id STRING, 
    total_spend_in_dollars FLOAT, 
    channel_spend MAP<STRING, FLOAT>, 
    campaign_start_date TIMESTAMP)
  USING DELTA
  LOCATION '{}'
  '''.format(gold_ad_spend_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3: Create widget for specifying the ad spend for a given campaign

# COMMAND ----------

dbutils.widgets.text("adspend", "10000", "Campaign Budget in $")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.4: Populate ad spend table with synthetic ad spend data

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO TABLE gold_ad_spend
# MAGIC VALUES ("03d65f7e92e81480cac52a20d", $adspend,
# MAGIC           MAP('Facebook', .2,
# MAGIC               'Instagram', .2,  
# MAGIC               'Online Display', .2, 
# MAGIC               'Paid Search', .2, 
# MAGIC               'Online Video', .2), 
# MAGIC          make_timestamp(2018, 7, 31, 0, 0, 0));

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.5: View campaign ad spend details

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_ad_spend

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.6: Explode struct into multiple rows

# COMMAND ----------

ad_spend_df = spark.sql('select explode(channel_spend) as (channel, pct_spend), \
                         round(total_spend_in_dollars * pct_spend, 2) as dollar_spend \
                         from gold_ad_spend')

ad_spend_df.createOrReplaceTempView("exploded_gold_ad_spend")
display(ad_spend_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.7: Base conversion rate

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE base_conversion_rate
# MAGIC USING DELTA AS
# MAGIC SELECT count(*) as count,
# MAGIC   CASE 
# MAGIC     WHEN conversion == 0 
# MAGIC     THEN 'Impression'
# MAGIC     ELSE 'Conversion'
# MAGIC   END AS interaction_type
# MAGIC FROM
# MAGIC   gold_user_journey
# MAGIC GROUP BY
# MAGIC   conversion;

# COMMAND ----------

base_converion_rate_pd = spark.table("base_conversion_rate").toPandas()

pie, ax = plt.subplots(figsize=[20,9])
labels = base_converion_rate_pd['interaction_type']
plt.pie(x=base_converion_rate_pd['count'], autopct="%.1f%%", explode=[0.05]*2, labels=labels, pctdistance=0.5)
plt.title("Base Conversion Rate");

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.8: Conversions by date

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE conversions_by_date 
# MAGIC USING DELTA AS
# MAGIC SELECT count(*) AS count,
# MAGIC   'Conversion' AS interaction_type,
# MAGIC   date(time) AS date
# MAGIC FROM bronze
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY date
# MAGIC ORDER BY date;

# COMMAND ----------

conversions_by_date_pd = spark.table("conversions_by_date").toPandas()

plt.figure(figsize=(20,9))
pt = sns.lineplot(x='date',y='count',data=conversions_by_date_pd)

pt.tick_params(labelsize=15)
pt.set_xlabel('Date')
pt.set_ylabel('Number of Conversions')
plt.title("Conversions by Date");

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.9: Attribution by model type

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE attribution_by_model_type 
# MAGIC USING DELTA AS
# MAGIC SELECT attribution_model, channel, round(attribution_percent * (
# MAGIC     SELECT count(*) FROM gold_user_journey WHERE conversion = 1)) AS conversions_attributed
# MAGIC FROM gold_attribution;

# COMMAND ----------

attribution_by_model_type_pd = spark.table("attribution_by_model_type").toPandas()

pt = sns.catplot(x='channel',y='conversions_attributed',hue='attribution_model',data=attribution_by_model_type_pd, kind='bar', aspect=4, legend=True)
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)

plt.tick_params(labelsize=15)
plt.ylabel("Number of Conversions")
plt.xlabel("Channels")
plt.title("Channel Performance");

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.10: Cost per acquisition

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE exploded_gold_ad_spend
# MAGIC USING DELTA AS
# MAGIC SELECT explode(channel_spend) AS (channel, spend),
# MAGIC    round(total_spend_in_dollars * spend, 2) AS dollar_spend
# MAGIC    FROM gold_ad_spend;

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE cpa_summary 
# MAGIC USING DELTA AS
# MAGIC SELECT
# MAGIC   spending.channel,
# MAGIC   spending.dollar_spend,
# MAGIC   attribution_count.attribution_model,
# MAGIC   attribution_count.conversions_attributed,
# MAGIC   round(spending.dollar_spend / attribution_count.conversions_attributed,2) AS CPA_in_Dollars
# MAGIC FROM
# MAGIC   (SELECT channel, dollar_spend
# MAGIC    FROM exploded_gold_ad_spend) AS spending
# MAGIC JOIN
# MAGIC   (SELECT attribution_model, channel, conversions_attributed
# MAGIC    FROM attribution_by_model_type) AS attribution_count
# MAGIC ON spending.channel = attribution_count.channel;

# COMMAND ----------

cpa_summary_pd = spark.table("cpa_summary").toPandas()

pt = sns.catplot(x='channel', y='CPA_in_Dollars',hue='attribution_model',data=cpa_summary_pd, kind='bar', aspect=4, ci=None)
plt.title("Cost of Aquisition by Channel")
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)

plt.tick_params(labelsize=15)
plt.ylabel("CPA in $")
plt.xlabel("Channels")
plt.title("Channel Cost per Aquisition");

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.11: Budget Allocation Optimization

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE spend_optimization_view 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT
# MAGIC   a.channel,
# MAGIC   a.pct_spend,
# MAGIC   b.attribution_percent,
# MAGIC   b.attribution_percent / a.pct_spend as ROAS,
# MAGIC   a.dollar_spend,
# MAGIC   round(
# MAGIC     (b.attribution_percent / a.pct_spend) * a.dollar_spend,
# MAGIC     2
# MAGIC   ) as proposed_dollar_spend
# MAGIC FROM
# MAGIC   exploded_gold_ad_spend a
# MAGIC   JOIN gold_attribution b on a.channel = b.channel
# MAGIC   and attribution_model = 'markov_chain';
# MAGIC   
# MAGIC CREATE
# MAGIC OR REPLACE TABLE spend_optimization_final 
# MAGIC USING DELTA AS
# MAGIC SELECT
# MAGIC   channel,
# MAGIC   'current_spending' AS spending,
# MAGIC   dollar_spend as budget
# MAGIC  FROM exploded_gold_ad_spend
# MAGIC UNION
# MAGIC SELECT
# MAGIC   channel,
# MAGIC   'proposed_spending' AS spending,
# MAGIC   proposed_dollar_spend as budget
# MAGIC FROM
# MAGIC   spend_optimization_view;  

# COMMAND ----------

spend_optimization_final_pd = spark.table("spend_optimization_final").toPandas()

pt = sns.catplot(x='channel', y='budget', hue='spending', data=spend_optimization_final_pd, kind='bar', aspect=4, ci=None)

plt.tick_params(labelsize=15)
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Spend Optimization per Channel")
plt.ylabel("Budget in $")
plt.xlabel("Channels")
