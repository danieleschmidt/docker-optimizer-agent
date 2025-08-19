FROM node:24
COPY package.json .
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]