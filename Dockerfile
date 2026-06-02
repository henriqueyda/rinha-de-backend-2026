FROM golang:1.26.3-alpine AS builder

WORKDIR /app

COPY . .

RUN go build -o app .

RUN ./app preprocess

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/app .
COPY resources ./resources
COPY --from=builder /app/resources/normalization.json ./resources/
COPY --from=builder /app/resources/mcc_risk.json ./resources/
COPY --from=builder /app/resources/index.bin ./resources/

EXPOSE 8080

CMD ["./app"]