import boto3
import json
import os

class AWSCareerService:
    def __init__(self, region="us-east-1"):
        self.s3 = boto3.client('s3', region_name=region)
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        self.bucket_name = "nexgen-career-agent-storage"

    def ensure_bucket_exists(self):
        """Creates the S3 bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists.")
        except Exception:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket {self.bucket_name} created.")
            except Exception as e:
                print(f"Error creating bucket: {e}")

    def upload_to_s3(self, local_path, s3_key):
        """Uploads a file to S3."""
        try:
            self.s3.upload_file(local_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except Exception as e:
            print(f"Upload error: {e}")
            return None

    def get_bedrock_analysis(self, resume_text):
        """GenAI: Skill-Gap Counselor Agent using AWS Bedrock (Claude 3.5 Sonnet)"""
        try:
            prompt = f"Human: Analyze this resume and recommend specific Cloud/Data certifications (AWS, Azure, GCP, Databricks). Also provide a 6-month study plan.\n\nResume Content:\n{resume_text}\n\nAssistant:"
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.5,
            })

            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0', # Using Claude 3.5 Sonnet
                body=body
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('content')[0].get('text')
        except Exception as e:
            return f"Bedrock Error: {str(e)}"

# Singleton instance
career_aws = AWSCareerService()
