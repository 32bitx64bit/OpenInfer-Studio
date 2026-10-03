package mediagen

import (
	"context"
	"crypto/subtle"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

// sdGateway is the authenticated managed endpoint for a stock sd-server.
// It has exactly one fixed loopback destination and no general proxy API.
type sdGateway struct {
	listener  net.Listener
	server    *http.Server
	transport *http.Transport
	cancel    context.CancelFunc
	once      sync.Once
	done      chan struct{}
}

func startSDGateway(listener net.Listener, nativePort int, key string) (*sdGateway, error) {
	if listener == nil {
		return nil, errors.New("invalid sd-server gateway endpoint")
	}
	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok || !addr.IP.Equal(net.IPv4(127, 0, 0, 1)) || addr.Port == 0 || nativePort < 1 || nativePort > 65535 || addr.Port == nativePort || key == "" {
		return nil, errors.New("invalid sd-server gateway endpoint")
	}
	targetAddr := net.JoinHostPort("127.0.0.1", strconv.Itoa(nativePort))
	target := &url.URL{Scheme: "http", Host: targetAddr}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	dialer := &net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if network != "tcp" || address != targetAddr {
			return nil, errors.New("invalid sd-server gateway destination")
		}
		return dialer.DialContext(ctx, "tcp4", targetAddr)
	}
	proxy := &httputil.ReverseProxy{
		Transport: transport,
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			r.Out.Host = targetAddr
			r.Out.URL.User = nil
			r.Out.URL.Opaque = ""
			r.Out.Header.Del("Authorization")
			r.Out.Header.Del("Proxy-Authorization")
			r.Out.Header.Del("Cookie")
			r.Out.Trailer = nil
		},
		ModifyResponse: func(resp *http.Response) error {
			if resp.StatusCode >= 300 && resp.StatusCode < 400 || resp.StatusCode == http.StatusSwitchingProtocols {
				return errors.New("sd-server gateway refuses redirects and upgrades")
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			http.Error(w, "sd-server unavailable", http.StatusBadGateway)
		},
		ErrorLog: log.New(io.Discard, "", 0),
	}
	ctx, cancel := context.WithCancel(context.Background())
	g := &sdGateway{listener: listener, transport: transport, cancel: cancel, done: make(chan struct{})}
	expected := []byte("Bearer " + key)
	g.server = &http.Server{
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      10 * time.Minute,
		IdleTimeout:       30 * time.Second,
		MaxHeaderBytes:    64 << 10,
		ErrorLog:          log.New(io.Discard, "", 0),
		BaseContext:       func(net.Listener) context.Context { return ctx },
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodConnect || r.Header.Get("Upgrade") != "" {
				http.Error(w, "method not supported", http.StatusMethodNotAllowed)
				return
			}
			keys := r.Header.Values("Authorization")
			if len(keys) != 1 || subtle.ConstantTimeCompare([]byte(keys[0]), expected) != 1 {
				w.Header().Set("WWW-Authenticate", "Bearer")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if !strings.HasPrefix(r.URL.Path, "/sdcpp/v1/") && !strings.HasPrefix(r.URL.Path, "/v1/") && !strings.HasPrefix(r.URL.Path, "/sdapi/v1/") {
				http.NotFound(w, r)
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, 128<<20)
			// Native model upscaling can take minutes. Bound the adapter's
			// lifetime while preserving the caller's shorter cancellation.
			requestCtx, requestCancel := context.WithTimeout(r.Context(), 10*time.Minute)
			defer requestCancel()
			proxy.ServeHTTP(w, r.WithContext(requestCtx))
		}),
	}
	go func() {
		defer close(g.done)
		_ = g.server.Serve(listener)
	}()
	return g, nil
}

// Close is idempotent and cancels active upstream requests as well as
// closing the managed listener. It never waits for native GPU work.
func (g *sdGateway) Close() {
	if g == nil {
		return
	}
	g.once.Do(func() {
		g.cancel()
		_ = g.server.Close()
		_ = g.listener.Close()
		g.transport.CloseIdleConnections()
	})
}
