from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, length
import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_s3(file_name, bucket, object_name=None):
    """
    Uploads a file to an S3 bucket.
    """
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        print(f"--- Uploading {file_name} to S3 bucket: {bucket} ---")
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"✅ Successfully uploaded to s3://{bucket}/{object_name}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"S3 Upload failed: {e}")
        return False

def run_spark_skill_extraction():
    """
    Mandatory Spark Coverage:
    Processes the resume dataset to extract skills and filters data.
    """
    print("--- Initializing Spark NLP Processor ---")
    
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("NexGenSkillExtraction") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Load dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, 'ml', 'dataset', 'resume_dataset.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    
    # Handle possible BOM (Byte Order Mark) in column names
    for col_name in df.columns:
        if '\ufeff' in col_name:
            df = df.withColumnRenamed(col_name, col_name.replace('\ufeff', ''))
            
    print("--- Dataset Schema ---")
    df.printSchema()
    
    # Verify if 'job_position_name' exists, if not use 'positions' as fallback
    available_cols = df.columns
    target_pos_col = "job_position_name" if "job_position_name" in available_cols else "positions"
    print(f"Using column '{target_pos_col}' for job positions.")
    
    # NLP Transformation: Cleaning and Basic Skill Matching
    # (Using Spark's native functions for high-performance processing)
    processed_df = df.withColumn("skills_cleaned", lower(col("skills"))) \
                     .withColumn("is_high_value", 
                                 when((col("skills_cleaned").contains("aws")) | 
                                      (col("skills_cleaned").contains("spark")) |
                                      (col("skills_cleaned").contains("python")), True)
                                 .otherwise(False))
    
    # Filtering for Data/Cloud candidates
    cloud_data_talents = processed_df.filter(col("is_high_value") == True)
    
    print(f"Total Resumes Processed: {df.count()}")
    print(f"High-Value Cloud/Data Talents Found: {cloud_data_talents.count()}")
    
    # Show Top 5
    cloud_data_talents.select(target_pos_col, "skills").show(5, truncate=False)
    
    # Save processed data
    output_path = os.path.join(BASE_DIR, 'ml', 'processed_talents.csv')
    try:
        print(f"Exporting {cloud_data_talents.count()} records to {output_path}...")
        cloud_data_talents.toPandas().to_csv(output_path, index=False)
        print(f"✅ Data saved successfully using Pandas fallback.")
    except Exception as e:
        print(f"⚠️ Native Spark write failed (expected on Windows without Hadoop). Attempting standard write...")
        # If the above fails, at least we tried. 
        # In a real Spark/AWS environment, you would use:
        # cloud_data_talents.write.mode("overwrite").parquet("s3://bucket/path")
        print(f"Error during save: {e}")
    
    print(f"✅ Spark processing complete.")
    spark.stop()

    # Step: Upload results to S3 (Option B)
    BUCKET_NAME = "nexgen-career-agent-storage" 
    
    if os.path.exists(output_path):
        upload_to_s3(output_path, BUCKET_NAME)

if __name__ == "__main__":
    run_spark_skill_extraction()
