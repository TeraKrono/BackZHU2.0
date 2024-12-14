#!/usr/bin/env bash

host="$1"
port="$2"
shift 2
cmd="$@"

until nc -z "$host" "$port"; do
  >&2 echo "Server $host:$port is unavailable - waiting"
  sleep 1
done

>&2 echo "Server $host:$port is up - executing command"
exec $cmd
