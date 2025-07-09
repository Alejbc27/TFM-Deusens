# Stage 1: Install dependencies and build the app
FROM node:20-slim AS builder
WORKDIR /app

# Copy package.json and lock file
COPY package*.json ./

# Install dependencies
RUN npm install
RUN npm audit fix

# Copy the rest of the application source code
COPY . .

# Build the Next.js application
RUN npm run build

# Stage 2: Create the production image
FROM node:20-slim AS runner
WORKDIR /app

# Set environment to production
ENV NODE_ENV=production

# Copy built files from the builder stage
COPY --from=builder /app/src ./src
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expose the port the app runs on
EXPOSE 3100

# Command to run the app
CMD ["npm", "start", "--", "-p", "3100"]
