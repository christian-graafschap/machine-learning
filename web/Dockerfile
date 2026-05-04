FROM nginx:alpine

# verwijder default nginx website
RUN rm -rf /usr/share/nginx/html/*

# kopieer jouw frontend
COPY . /usr/share/nginx/html

# expose is optioneel (NGINX gebruikt standaard 80)
EXPOSE 80