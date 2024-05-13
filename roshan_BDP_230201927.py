import sys
import string
import os
import socket
import time
import operator
import boto3
import json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import dense_rank, avg, dayofmonth
from pyspark.sql.functions import desc, rank, col, concat, lit, sum, when
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import from_unixtime, count, sum, month
from pyspark.sql.window import Window
from graphframes import GraphFrame

if __name__ == "__main__":
    # Initialize SparkSession
    spark = SparkSession.builder.appName("Rideshare").getOrCreate()

    # Function to validate data lines
    def is_valid_line(line):
        try:
            fields = line.split(',')
            if len(fields) != 9:
                return False
            float(fields[6])
            float(fields[7])
            return True
        except:
            return False

    # Accessing environment variables for AWS credentials and bucket details
    s3_data_repository_bucket = os.environ['DATA_REPOSITORY_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL'] + ':' + os.environ['BUCKET_PORT']
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['BUCKET_NAME']

    # Configure Hadoop for S3 access
    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.endpoint", s3_endpoint_url)
    hadoopConf.set("fs.s3a.access.key", s3_access_key_id)
    hadoopConf.set("fs.s3a.secret.key", s3_secret_access_key)
    hadoopConf.set("fs.s3a.path.style.access", "true")
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "false")

    # Task 1 Part I: Load rideshare_data and taxi_zone_lookup tables from shared bucket
    rideshare_data_path = f"s3a://{s3_data_repository_bucket}/ECS765/rideshare_2023/rideshare_data.csv"
    rideshare_data = spark.read.format("csv").option("header", "true").load(rideshare_data_path)
   
    taxi_zone_lookup_path = f"s3a://{s3_data_repository_bucket}/ECS765/rideshare_2023/taxi_zone_lookup.csv"
    taxi_zone_lookup = spark.read.format("csv").option("header", "true").load(taxi_zone_lookup_path)
    
    # Print the first few rows of each DataFrame
    #print("Rideshare Data:")
    rideshare_data.show(5)
    
    #print("Taxi Zone Lookup:")
    taxi_zone_lookup.show(5)
    
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

    print("DataFrame after join and column renaming:")
    val2.show(5)
    val2.printSchema()


    # Task 1 Part III: Convert UNIX timestamp to "yyyy-MM-dd" format
    val2 = val2.withColumn("date", from_unixtime("date").cast("date"))
    # Print the output after converting UNIX timestamp to "yyyy-MM-dd" format
    
    print("DataFrame after converting UNIX timestamp to 'yyyy-MM-dd' format:")
    val2.printSchema()

     
       
     # Task 1 Part IV: Print the number of rows and schema of the new dataframe
    print(f"\n")
    print(f"\n")
    print(f"\n")
    print("Number of Rows:", val2.count())
    print(f"\n")
    print(f"\n")
    print(f"\n")


    # Task 2 - Part i
    # Count the number of trips for each business in each month
    trips_with_month = val2.withColumn("month", month("date"))
    trip_counts = trips_with_month.groupBy("business", "month").agg(count("*").alias("trip_count"))

    # to print trip counts
    trip_counts.show()

    #downloading the data in csv
    import pandas as pd
    import boto3
    from io import StringIO
    from datetime import datetime

    
    # Convert trip_counts DataFrame to Pandas DataFrame
    trip_counts_df_pd = trip_counts.toPandas()
    
    # Get current date and time for timestamp
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # CSV file name with timestamp
    csv_filename = f'trip_counts_{date_time}.csv'
    
    # Convert Pandas DataFrame to CSV format in memory
    csv_buffer = StringIO()
    trip_counts_df_pd.to_csv(csv_buffer, index=False)
    
    # Upload CSV file to S3 bucket
    s3 = boto3.client('s3',
                      endpoint_url=f'http://{s3_endpoint_url}',
                      aws_access_key_id=s3_access_key_id,
                      aws_secret_access_key=s3_secret_access_key)
    s3.put_object(Bucket=s3_bucket, Key=csv_filename, Body=csv_buffer.getvalue())
    
    # Download the CSV file from S3
    s3.download_file(s3_bucket, csv_filename, csv_filename)
    
    print(f'CSV file "{csv_filename}" saved in S3 bucket "{s3_bucket}" and downloaded locally.')
    
    

    # Task 2 - Part ii
    # Calculate the platform's profits for each business in each month
    profits_with_month = val2.withColumn("month", month("date"))
    profits_by_business_month = profits_with_month.groupBy("business", "month") \
        .agg(sum("rideshare_profit").cast("float").alias("total_profit"))

    # print the profits by business and month
    profits_by_business_month.show()

    #downloading the data in csv
    
    
    # Convert profits_by_business_month DataFrame to Pandas DataFrame
    profits_by_business_month_pd_df = profits_by_business_month.toPandas()
    
    # Get current date and time for timestamp
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # CSV file name with timestamp
    csv_filename = f'profits_by_business_month_{date_time}.csv'
    
    # Convert Pandas DataFrame to CSV format in memory
    csv_buffer = StringIO()
    profits_by_business_month_pd_df.to_csv(csv_buffer, index=False)
    
    # Upload CSV file to S3 bucket
    s3 = boto3.client('s3',
                      endpoint_url=f'http://{s3_endpoint_url}',
                      aws_access_key_id=s3_access_key_id,
                      aws_secret_access_key=s3_secret_access_key)
    s3.put_object(Bucket=s3_bucket, Key=csv_filename, Body=csv_buffer.getvalue())
    
    # Download the CSV file from S3
    s3.download_file(s3_bucket, csv_filename, csv_filename)
    
    print(f'CSV file "{csv_filename}" saved in S3 bucket "{s3_bucket}" and downloaded locally.')
    
   
    # task 2 - part iii
     # Assigning a new DataFrame with the "driver_total_pay" column converted to float
    earnings_with_float = val2.withColumn("driver_total_pay", val2["driver_total_pay"].cast("float"))

    # Assigning a new DataFrame with the month extracted from the date column
    earnings_with_month = earnings_with_float.withColumn("month", month("date"))

    # Group by business and month to sum the driver total pay
    earnings_by_business_month = earnings_with_month.groupBy("business", "month") \
        .agg(sum("driver_total_pay").cast("float").alias("total_earnings"))
    
     # Display the earnings by business and month

    earnings_by_business_month.show()
    
     #downloading the data in csv
    import pandas as pd
    import boto3
    from io import StringIO
    from datetime import datetime
    
    
    # Convert earnings_by_business_month DataFrame to Pandas DataFrame
    earnings_by_business_month_pd_df= earnings_by_business_month.toPandas()
    
    # Get current date and time for timestamp
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # CSV file name with timestamp
    csv_filename = f'earnings_by_business_month_{date_time}.csv'
    
    # Convert Pandas DataFrame to CSV format in memory
    csv_buffer = StringIO()
    earnings_by_business_month_pd_df.to_csv(csv_buffer, index=False)
    
    # Upload CSV file to S3 bucket
    s3 = boto3.client('s3',
                      endpoint_url=f'http://{s3_endpoint_url}',
                      aws_access_key_id=s3_access_key_id,
                      aws_secret_access_key=s3_secret_access_key)
    s3.put_object(Bucket=s3_bucket, Key=csv_filename, Body=csv_buffer.getvalue())
    
    # Download the CSV file from S3
    s3.download_file(s3_bucket, csv_filename, csv_filename)
    
    print(f'CSV file "{csv_filename}" saved in S3 bucket "{s3_bucket}" and downloaded locally.')

    
    # Task 3 Part - i:
    #Identify the top 5 popular pickup boroughs each month
    top_pickup_boroughs = val2.groupBy("Pickup_Borough", month("date").alias("Month")) \
        .agg(count("*").alias("trip_count")) \
        .orderBy("Month", desc("trip_count"))
    
    window_spec = Window.partitionBy("Month").orderBy(desc("trip_count"))
    
    top_pickup_boroughs_with_rank = top_pickup_boroughs.withColumn("rank", rank().over(window_spec))
    
    top_5_pickup_boroughs_each_month = top_pickup_boroughs_with_rank.filter(col("rank") <= 5) \
        .drop("rank") \
        .orderBy("Month", desc("trip_count"))
    
    # Show the top 5 pickup boroughs for each month
    top_5_pickup_boroughs_each_month.show(25)



 
    # Task 3 - Part ii
    # Step 1: Group by dropoff borough and month, and count the number of trips
    dropoff_counts = val2.groupBy("Dropoff_Borough", month("date").alias("Month")).agg(count("*").alias("trip_count"))

    # Step 2: Rank the dropoff boroughs within each month based on trip count
    window_spec = Window.partitionBy("Month").orderBy(desc("trip_count"))
    dropoff_counts = dropoff_counts.withColumn("rank", dense_rank().over(window_spec))

    # Step 3: Sort the output by trip count by descending order within each month
    dropoff_counts = dropoff_counts.orderBy("Month", desc("trip_count"))

    # Step 4: Filter the top 5 dropoff boroughs for each month
    top_dropoff_boroughs = dropoff_counts.filter(col("rank") <= 5).select("Dropoff_Borough", "Month", "trip_count")
    
    # Show the result
    top_dropoff_boroughs.show(25)


    
    # Task 3 - Part iii
    # Identify the top 30 earnest routes
    top_30_routes = val2.groupBy(concat(col("Pickup_Borough"), lit(" to "), col("Dropoff_Borough")).alias("Route")) \
                    .agg(sum("driver_total_pay").alias("total_profit")) \
                    .orderBy(desc("total_profit"))

    # Show the top 30 earnest routes

    top_30_routes.show(30)
    

    #task 4 part -1

    #Calculate the average 'driver_total_pay' during different 'time_of_day' periods

    average_pay_by_time_of_day = val2.groupBy("time_of_day") \
    .agg(avg("driver_total_pay").alias("average_driver_total_pay")) \
    .orderBy(desc("average_driver_total_pay"))

    # Show the result

    average_pay_by_time_of_day.show(truncate=False)


    #task 4 part - ii

    # Group by 'time_of_day' and calculate the average 'trip_length'

    avg_trip_length_by_time_of_day = val2.groupBy("time_of_day").agg(avg("trip_length").alias("average_trip_length"))

    # Sort the output by average_trip_length in descending order
    sorted_avg_trip_length = avg_trip_length_by_time_of_day.orderBy("average_trip_length", ascending=False)

    # Show the result
    
    sorted_avg_trip_length.show()
    

    #task 4 part iii
    # Task 4 - Calculate the average 'driver_total_pay' during different 'time_of_day' periods
    average_pay_by_time_of_day = val2.groupBy("time_of_day") \
    .agg(avg("driver_total_pay").alias("average_driver_total_pay")) \
    .orderBy(desc("average_driver_total_pay"))

    # Group by 'time_of_day' and calculate the average 'trip_length'
    avg_trip_length_by_time_of_day = val2.groupBy("time_of_day").agg(avg("trip_length").alias("average_trip_length"))

    # Sort the output by average_trip_length in descending order
    sorted_avg_trip_length = avg_trip_length_by_time_of_day.orderBy("average_trip_length", ascending=False)

    # Join the DataFrames sorted_avg_trip_length and average_pay_by_time_of_day based on 'time_of_day'
    joined_data = sorted_avg_trip_length.join(average_pay_by_time_of_day, "time_of_day", "inner")

    # Calculate the average earning per mile
    avg_earning_per_mile = joined_data.withColumn("average_earning_per_mile", col("average_driver_total_pay") / col("average_trip_length"))

    # Select only the required columns
    result = avg_earning_per_mile.select("time_of_day", "average_earning_per_mile")

    # Show the result
    result.show()

    
    #task 5 part - i

    # Filter data to include only records from January

    january_data = val2.filter(month(val2["date"]) == 1)

    average_waiting_time_per_day = january_data.groupBy(dayofmonth("date").alias("day"))

    # Calculate the average waiting time for each day in January

    average_waiting_time_per_day = average_waiting_time_per_day.agg(avg("request_to_pickup").alias("average_waiting_time"))

    # Sort the output by day

    sorted_average_waiting_time = average_waiting_time_per_day.orderBy("day")

    # Show the result
    sorted_average_waiting_time.show(10)

    #downloading the data in csv
    import pandas as pd
    import boto3
    from io import StringIO
    from datetime import datetime

    #downloading the data in csv
    import pandas as pd
    import boto3
    from io import StringIO
    from datetime import datetime
    
    # Convert sorted_avg_waiting_time_pd DataFrame to Pandas DataFrame
    sorted_avg_waiting_time_pd_df =  sorted_average_waiting_time.toPandas()
    
    # Get current date and time for timestamp
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # CSV file name with timestamp
    csv_filename = f'sorted_avg_waiting_time_{date_time}.csv'
    
    # Convert Pandas DataFrame to CSV format in memory
    csv_buffer = StringIO()
    sorted_avg_waiting_time_pd_df.to_csv(csv_buffer, index=False)
    
    # Upload CSV file to S3 bucket
    s3 = boto3.client('s3',
                      endpoint_url=f'http://{s3_endpoint_url}',
                      aws_access_key_id=s3_access_key_id,
                      aws_secret_access_key=s3_secret_access_key)
    s3.put_object(Bucket=s3_bucket, Key=csv_filename, Body=csv_buffer.getvalue())
    
    # Download the CSV file from S3
    s3.download_file(s3_bucket, csv_filename, csv_filename)
    
    print(f'CSV file "{csv_filename}" saved in S3 bucket "{s3_bucket}" and downloaded locally.')
    
    

    
            
    #task 5 part-ii
        
    # Filter the DataFrame to find days with average waiting time exceeding 300 seconds
    days_exceeding_300 = sorted_average_waiting_time.filter(col("average_waiting_time") > 300)
    
    # Show the result
    
    days_exceeding_300.show()

    

    #task 6 part- i
    
        # Calculate the trip count by grouping 'Pickup_Borough' and 'time_of_day'
    trip_count_by_pickup_borough_time_of_day = val2.groupBy("Pickup_Borough", "time_of_day") \
        .agg(count("*").alias("trip_count"))
    
    # Filter the DataFrame for trip counts greater than 0 and less than 1000
    filtered_trips = trip_count_by_pickup_borough_time_of_day.filter((col("trip_count") > 0) & (col("trip_count") < 1000))
    
    # Show the result
    filtered_trips.show()


    #task 6 part -ii
    # Filter the DataFrame to include only evening trips
    evening_trips = val2.filter(col("time_of_day") == "evening")
    
    # Group by 'Pickup_Borough' and aggregate the trip counts
    evening_trip_counts = evening_trips.groupBy("Pickup_Borough", "time_of_day") \
        .agg(count("*").alias("trip_count"))
    
    # Show the result
    evening_trip_counts.show()

    #task6 -part iii
    # Filter the DataFrame to include only trips from Brooklyn to Staten Island
    brooklyn_to_staten_island_trips = val2.filter((col("Pickup_Borough") == "Brooklyn") & (col("Dropoff_Borough") == "Staten Island"))
    
    # Select the required columns
    selected_columns = brooklyn_to_staten_island_trips.select("Pickup_Borough", "Dropoff_Borough", "Pickup_Zone")
    
    # Show 10 samples
    selected_columns.show(10, truncate=False)

    

    #task 7 
    # Create a new column 'Route' by concatenating 'Pickup_Zone' with 'to' and 'Dropoff_Zone'
    routes_df = val2.withColumn('Route', concat(col('Pickup_Zone'), lit(' to '), col('Dropoff_Zone')))
    
    # Group by 'Route' and aggregate trip counts for Uber and Lyft separately
    route_counts = routes_df.groupBy('Route') \
        .agg(sum(when(col('business') == 'Uber', 1).otherwise(0)).alias('uber_count'),
             sum(when(col('business') == 'Lyft', 1).otherwise(0)).alias('lyft_count'))
    
    # Calculate total count for each route
    route_counts = route_counts.withColumn('total_count', col('uber_count') + col('lyft_count'))
    
    # Sort the DataFrame by total count in descending order
    sorted_route_counts = route_counts.orderBy(desc('total_count'))
    
    # Select the top 10 routes based on total count
    top_10_routes = sorted_route_counts.select('Route', 'uber_count', 'lyft_count', 'total_count').limit(10)
    
    # Show the result
    top_10_routes.show(truncate=False)
    
# Stop SparkSession
spark.stop()

    








    
    
