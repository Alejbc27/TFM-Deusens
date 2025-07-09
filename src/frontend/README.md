# NeonNexus Chat


A visually engaging, futuristic chat UI powered by Next.js, featuring:
- Neon chat interface with modern, glowing design
- Message history
- Docker support for easy deployment

## Prerequisites
- [Node.js](https://nodejs.org/) v18 or later (Node 20 recommended)
- [npm](https://www.npmjs.com/) v9 or later

## Installation
Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd agent-frontend
npm install
```

## Development
### 1. Start the Next.js Frontend
```bash
npm run dev
```
This starts the app at [http://localhost:9002](http://localhost:9002).



### 3. Start the Agent Backend (Required)
The chat UI expects an agent backend running at [http://localhost:8081](http://localhost:8081). Ensure you have the backend running separately. The frontend sends chat messages as JSON to `/chat` on this backend.

## Build for Production
```bash
npm run build
```

## Start in Production Mode
```bash
npm start
```
By default, the app runs on port 3000 in production (unless overridden by the environment or Docker).

## Docker
Build and run the app using Docker:
```bash
docker build -t neon-nexus-chat .
docker run -p 3100:3100 neon-nexus-chat
```

> **Note:** The Docker container exposes port 3100 (instead of 3000) to avoid conflict with Grafana, which uses port 3000 by default.

## Environment Variables
- The project loads environment variables from a `.env` file using `dotenv`, but no required variables are currently enforced. You may add your own as needed.

## Troubleshooting
- If you see errors connecting to the agent, ensure the backend is running at `http://localhost:8081`.
- For style or font issues, check your internet connection for Google Fonts.

## License
MIT 