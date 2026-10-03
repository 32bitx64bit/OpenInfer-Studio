package mediagen

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
)

func sdHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil // a loopback process must never use environment proxies
	return &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

// doSD only reaches a supervised loopback process and never redirects its
// bearer credential to another listener. The key is kept solely in memory.
func (m *Manager) doSD(req *http.Request, port int) (*http.Response, error) {
	if req.URL.Scheme != "http" || req.URL.Host != "127.0.0.1:"+strconv.Itoa(port) || req.URL.User != nil || req.URL.Opaque != "" || port < 1 || port > 65535 {
		return nil, fmt.Errorf("invalid sd-server loopback address")
	}
	m.mu.Lock()
	var found bool
	var key string
	for _, sv := range m.servers {
		if sv.port == port && port != 0 {
			found = true
			key = sv.apiKey
			break
		}
	}
	m.mu.Unlock()
	if !found {
		return nil, ErrServerNotRunning
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	// Copy the client to disallow redirects even in tests/custom transports.
	client := *m.http
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return client.Do(req)
}

func sdProcessAuth(help string) (string, error) {
	// Every managed endpoint has a fresh in-memory key, including stock
	// runtimes whose authentication is supplied by the local gateway.
	var key [32]byte
	if _, err := rand.Read(key[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(key[:]), nil
}

func advertisesSDOption(help, option string) bool {
	if removedInHelp(help, option) {
		return false
	}
	for _, line := range strings.Split(help, "\n") {
		// Only an option declaration counts, never prose mentioning a flag.
		for _, field := range strings.Fields(strings.TrimSpace(line)) {
			field = strings.Trim(field, ",")
			if !strings.HasPrefix(field, "-") {
				break
			}
			name, _, _ := strings.Cut(field, "=")
			if name == option {
				return true
			}
		}
	}
	return false
}

// sdEndpoint owns the public authenticated endpoint for one launch. For a
// stock runtime the native port is internal loopback transport, never the
// port exposed through ServerPort, events, or the manager's HTTP client.
type sdEndpoint struct {
	port, nativePort int
	key              string
	gateway          *sdGateway
}

func (e *sdEndpoint) close() {
	if e != nil && e.gateway != nil {
		e.gateway.Close()
	}
}

func configureSDTransport(help string, args []string) ([]string, *sdEndpoint, error) {
	key, err := sdProcessAuth(help)
	if err != nil {
		return nil, nil, err
	}
	e := &sdEndpoint{key: key}
	nativeAuth := advertisesSDOption(help, "--api-key")
	var listener net.Listener
	if !nativeAuth {
		// Keep the public socket bound while selecting the separate native
		// port, so the two endpoints cannot accidentally be the same.
		listener, err = net.Listen("tcp4", "127.0.0.1:0")
		if err != nil {
			return nil, nil, err
		}
		e.port = listener.Addr().(*net.TCPAddr).Port
	}
	e.nativePort, err = allocatePort()
	if err != nil {
		if listener != nil {
			_ = listener.Close()
		}
		return nil, nil, err
	}
	out := append([]string(nil), args...)
	patched := false
	for i := 0; i < len(out); i++ {
		if out[i] == "--api-key" || out[i] == "--api-key-file" || strings.HasPrefix(out[i], "--api-key=") || strings.HasPrefix(out[i], "--api-key-file=") {
			if listener != nil {
				_ = listener.Close()
			}
			return nil, nil, fmt.Errorf("sd-server authentication arguments must be managed")
		}
		if out[i] == "--listen-port" && i+1 < len(out) {
			out[i+1] = strconv.Itoa(e.nativePort)
			patched = true
		}
	}
	if !patched {
		if listener != nil {
			_ = listener.Close()
		}
		return nil, nil, fmt.Errorf("sd-server requires an advertised --listen-port option")
	}
	if nativeAuth {
		e.port = e.nativePort
		out = append(out, "--api-key", key)
	} else {
		e.gateway, err = startSDGateway(listener, e.nativePort, key)
		if err != nil {
			_ = listener.Close()
			return nil, nil, err
		}
	}
	return out, e, nil
}

func redactSDArgs(args []string) []string {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] == "--api-key" {
			out[i+1] = "[redacted]"
		}
	}
	return out
}
