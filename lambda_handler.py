import json
import boto3
import base64

def lambda_handler(event, context):
    """
    AWS Lambda Handler for Resume Analysis.
    Triggered by S3 or API Gateway.
    """
    bedrock = boto3.client(service_name='bedrock-runtime')
    
    # Extract text from event (example: from API Gateway body)
    try:
        if 'body' in event:
            body = json.loads(event['body'])
            resume_text = body.get('resume_text', '')
        else:
            resume_text = event.get('resume_text', '')

        if not resume_text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No resume text provided'})
            }

        prompt = f"Human: Analyze this resume for cloud engineering roles and suggest certifications.\n\nResume:\n{resume_text}\n\nAssistant:"

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        })

        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=body
        )
        
        result = json.loads(response.get('body').read())
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'analysis': result['content'][0]['text'],
                'source': 'AWS Lambda + Bedrock'
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
