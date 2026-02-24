# Hosting Streamlit on AWS EC2 - Implementation Plan

This guide will walk you through the process of hosting your NexGen AI Agent System (Streamlit) on an AWS EC2 instance.

## Phase 1: AWS EC2 Setup
1. **Launch Instance**:
   - **AMI**: Ubuntu 22.04 LTS (Recommended).
   - **Instance Type**: t3.medium (Recommended for GenAI apps) or t2.micro (Free Tier, but might be slow).
   - **Key Pair**: Create or use an existing .pem key for SSH.
2. **Configure Security Group**:
   - Add **Inbound Rule**: Type: `Custom TCP`, Port: `8501`, Source: `0.0.0.0/0` (For Streamlit).
   - Add **Inbound Rule**: Type: `SSH`, Port: `22`, Source: `Your IP`.
   - Optional: Add **Inbound Rule**: Type: `HTTP`, Port: `80`, Source: `0.0.0.0/0` (If using Nginx).
3. **IAM Role**:
   - Attach an IAM role to the instance with permissions if needed for S3.

## Phase 2: Server Configuration
Connect to your instance: `ssh -i your-key.pem ubuntu@your-ec2-ip`

Run the following commands to set up the environment:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git nginx -y
```

## Phase 3: Deployment
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd resume_matching
   ```
2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   - Create a `.env` file in the root directory:
     ```bash
     nano .env
     ```
   - Paste your API key:
     ```text
     GEMINI_API_KEY=YOUR_SECURE_API_KEY_HERE
     ```

## Phase 4: Running the App (Production Style)
We will use `systemd` to keep the app running in the background and restart it on failure.

1. **Create Service File**:
   ```bash
   sudo nano /etc/systemd/system/streamlit_app.service
   ```
2. **Paste the following**:
   ```ini
   [Unit]
   Description=Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/resume_matching
   ExecStart=/home/ubuntu/resume_matching/venv/bin/streamlit run frontend/app.py --server.port 8501
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
3. **Start Service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start streamlit_app
   sudo systemctl enable streamlit_app
   ```

## Phase 5: Nginx Port Forwarding (Port 8501 -> Port 80)
To access the site via `http://your-ip` instead of `http://your-ip:8501`:
1. Edit Nginx config: `sudo nano /etc/nginx/sites-available/default`
2. Update the `location /` block:
   ```nginx
   location / {
       proxy_pass http://localhost:8501;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
       proxy_cache_bypass $http_upgrade;
   }
   ```
3. Restart Nginx: `sudo systemctl restart nginx`
