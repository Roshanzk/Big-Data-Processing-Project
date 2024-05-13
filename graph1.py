import sys, string
import os
import socket
import time
import operator
import boto3
import json
from pyspark.sql import SparkSession
from datetime import datetime

from functools import reduce
from pyspark.sql.functions import col, lit, when, desc
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
import graphframes
from graphframes import *

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
        .appName("graphframes") \
        .getOrCreate()

    sqlContext = SQLContext(spark)
    # shared read-only object bucket containing datasets
    s3_data_repository_bucket = os.environ['DATA_REPOSITORY_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL'] + ':' + os.environ['BUCKET_PORT']
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['BUCKET_NAME']

    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.endpoint", s3_endpoint_url)
    hadoopConf.set("fs.s3a.access.key", s3_access_key_id)
    hadoopConf.set("fs.s3a.secret.key", s3_secret_access_key)
    hadoopConf.set("fs.s3a.path.style.access", "true")
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "false")

    rideshare_data_path = f"s3a://{s3_data_repository_bucket}/ECS765/rideshare_2023/rideshare_data.csv"
    rideshare_data = spark.read.format("csv").option("header", "true").load(rideshare_data_path)
   
    taxi_zone_lookup_path = f"s3a://{s3_data_repository_bucket}/ECS765/rideshare_2023/taxi_zone_lookup.csv"
    taxi_zone_lookup = spark.read.format("csv").option("header", "true").load(taxi_zone_lookup_path) 
    
    # Task 1 Part II: Perform table join and rename columns
    val1 = rideshare_data.join(taxi_zone_lookup, rideshare_data.pickup_location == taxi_zone_lookup.LocationID, "left") \
        .select(
            [rideshare_data[col] for col in rideshare_data.columns] +
            [taxi_zone_lookup["Borough"].alias("Pickup_Borough"),
             taxi_zone_lookup["Zone"].alias("Pickup_Zone"),
             taxi_zone_lookup["service_zone"].alias("Pickup_service_zone")]
        )

    val2= val1.join(taxi_zone_lookup, val1.dropoff_location == taxi_zone_lookup.LocationID, "left") \
        .select(
            [val1[col] for col in val1.columns] +
            [taxi_zone_lookup["Borough"].alias("Dropoff_Borough"),
             taxi_zone_lookup["Zone"].alias("Dropoff_Zone"),
             taxi_zone_lookup["service_zone"].alias("Dropoff_service_zone")]
        ) 
        
    
    # Print the first few rows of the resulting dataframe after the join and column renaming

    # print("DataFrame after join and column renaming:")
    # val2.show(5)
    # val2.printSchema()


                   
    
    

    #task8 part - i
    # Define the schema for the vertex data (taxi_zone_lookup.csv)
    vertexSchema = StructType([
        StructField("LocationID", StringType(), True),
        StructField("Borough", StringType(), True),
        StructField("Zone", StringType(), True),
        StructField("service_zone", StringType(), True)
    ])
    
    # Define the schema for the edge data (rideshare_data.csv)
    edgeSchema = StructType([
        StructField("pickup_location", StringType(), True),
        StructField("dropoff_location", StringType(), True),
        StructField("trip_length", StringType(), True),
        StructField("request_to_pickup", StringType(), True),
        StructField("total_ride_time", StringType(), True),
        StructField("on_scene_to_pickup", StringType(), True),
        StructField("on_scene_to_dropoff", StringType(), True),
        StructField("time_of_day", StringType(), True),
        StructField("date", StringType(), True),
        StructField("passenger_fare", StringType(), True),
        StructField("driver_total_pay", StringType(), True),
        StructField("rideshare_profit", StringType(), True),
        StructField("hourly_rate", StringType(), True),
        StructField("dollars_per_mile", StringType(), True)
    ])
        # Print the vertexSchema
    print("Vertex Schema:")
    print(vertexSchema)
    print("\n")
    
        # Print the edgeSchema
    print("Edge Schema:")
    print(edgeSchema)
    
    #task 8 part -ii
        # Construct edges dataframe
    edges_df = val2.select("pickup_location", "dropoff_location").toDF("src", "dst")
    
    # Construct vertices dataframe
    vertices_df = taxi_zone_lookup.select("LocationID", "Borough", "Zone", "service_zone").toDF("id", "Borough", "Zone", "service_zone")
    
    # # Show 10 samples of edges dataframe

    print("Edges DataFrame:")
    edges_df.show(10, truncate=False)
    
    # # Show 10 samples of vertices dataframe

    print("Vertices DataFrame:")
    vertices_df.show(10, truncate=False)
    

    #task 8 part -iii
    
    # Create a GraphFrame instance

    graph = GraphFrame(vertices_df, edges_df)
    
    # Adjust the display settings to show the full content of each cell without truncation

    spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
    spark.conf.set("spark.sql.repl.eagerEval.maxNumRows", 20)
    
    # Print 10 samples of the graph DataFrame with columns ‘src’, ‘edge’, and ‘dst’

    graph.find("(src)-[edge]->(dst)").show(10, truncate=False)



    # Task 8 - Part iv

    #Count connected vertices with the same Borough and same service_zone

    connected_vertices_count = graph.triplets \
        .filter("src.Borough = dst.Borough and src.service_zone = dst.service_zone") \
        .groupBy("src.id", "dst.id", "src.Borough", "src.service_zone") \
        .count() \
        .orderBy("count", ascending=False)
    
    # Select 10 samples from the result
    connected_vertices_count.show(10)
    

    # Task 8 - Part v

    # Perform page ranking on the graph DataFrame
    page_rank_results = graph.pageRank(resetProbability=0.17, tol=0.01)
    
    # Sort vertices by descending order according to the value of PageRank
    sorted_page_rank = page_rank_results.vertices.select("id", "pagerank").orderBy("pagerank", ascending=False)
    
    # Show the top 5 samples of the results

    sorted_page_rank.show(5)

# Stop SparkSession
spark.stop()





