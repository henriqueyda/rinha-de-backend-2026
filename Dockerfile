FROM golang:1.26.3-alpine AS builder

WORKDIR /app

COPY . .

RUN go build -o app .

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/app .
COPY resources ./resources

EXPOSE 8080

CMD ["./app"]