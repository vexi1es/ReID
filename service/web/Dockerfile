FROM nginx:1.27-alpine
COPY service/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY service/web/ /usr/share/nginx/html/
RUN rm -f /usr/share/nginx/html/Dockerfile /usr/share/nginx/html/nginx.conf
