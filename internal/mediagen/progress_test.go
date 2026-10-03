package mediagen

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLogProgressDistinguishesWeightsFromSamplingAndHandlesPartialWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.log")
	if err := os.WriteFile(path, []byte("old job\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := newLogProgress(path, logSize(path))
	appendLog := func(s string) {
		t.Helper()
		f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		_, err = f.WriteString(s)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	appendLog("\r  |########  | 283/666 - 5.70GB/s\x1b[K")
	v := p.snapshot("job", "generating")
	if v.Unit != "tensors" || v.Current != 283 || v.Total != 666 || v.Phase != "loading_weights" {
		t.Fatalf("tensor counts misreported as sampling: %+v", v)
	}
	appendLog("\n[INFO] loading tensors completed, taking 1s\n\r  |=>  | 1/20 - 2.34s/")
	v = p.snapshot("job", "generating")
	if v.Total != 0 {
		t.Fatalf("completed weight counter remained visible: %+v", v)
	}
	appendLog("it\x1b[K")
	v = p.snapshot("job", "generating")
	if v.Unit != "steps" || v.Current != 1 || v.Total != 20 || v.Phase != "sampling" {
		t.Fatalf("partial sampling bar not reconstructed: %+v", v)
	}
	appendLog("\n\r |#### | 12/84 - 117.31MB/s\n")
	v = p.snapshot("job", "generating")
	if v.Unit != "tensors" || v.SamplingCurrent != 1 || v.SamplingTotal != 20 || !strings.Contains(v.Message, "Sampling: step 1/20;") {
		t.Fatalf("weight offloading hid completed sampling steps: %+v", v)
	}
	p.lastOutput = time.Now().Add(-time.Minute)
	v = p.snapshot("job", "generating")
	if v.QuietMS < 59000 || v.ServerResponding == nil || !*v.ServerResponding || v.SamplingCurrent != 1 {
		t.Fatalf("quiet server advanced progress without evidence: %+v", v)
	}
	v = p.snapshot("job", "unresponsive")
	if v.ServerResponding == nil || *v.ServerResponding {
		t.Fatal("failed HTTP poll claimed server was responding")
	}
	appendLog("\n/src/ggml-cuda/binbcast.cu:293: GGML_ASSERT(nb10 % sizeof(src1_t) == 0) failed\n")
	p.snapshot("job", "generating")
	if !strings.Contains(p.fatal, "nb10") || !strings.Contains(runtimeFailure(p.fatal, path), "tensor layout") {
		t.Fatal("runtime assertion was not identified")
	}
}

func TestRuntimeFailureKeepsAssertionAheadOfLongBarsAndBacktrace(t *testing.T) {
	log := "=== sd-server starting current ===\n" +
		strings.Repeat("\r |#### | 128/4349 - 500.0MB/s\x1b[K", 100) +
		"\n/src/ggml-cuda/binbcast.cu:293: GGML_ASSERT(nb10 % sizeof(src1_t) == 0) failed\n" +
		strings.Repeat("#0 0x01 in main ()\n", 100)
	msg := runtimeFailure(log, "/server.log")
	if !strings.Contains(msg, "tensor layout") || !strings.Contains(msg, "GGML_ASSERT") || strings.Contains(msg, "#0") || len(msg) > 500 {
		t.Fatalf("unhelpful crash report: %s", msg)
	}
}

func TestDecodeTilesAreNotReportedAsNewSamplingSteps(t *testing.T) {
	p := newLogProgress("", 0)
	p.line("|==========| 20/20 - 5.00s/it")
	p.line("[INFO] video.cpp:1712 - sampling completed, taking 77.98s")
	p.line("[INFO] ggml_graph_cut.cpp:1032 - ltx_video_vae build cached graph cut plan done (taking 1 ms)")
	p.line("|==========| 86/86 - 502.99MB/s")
	p.line("[INFO] model_loader.cpp:1378 - loading tensors completed, taking 1.54s")
	p.line("|=====>    | 1/10 - 4.17s/it")
	if v := p.progress; v.Phase != "decoding" || v.Unit != "tiles" || v.Current != 1 || v.Total != 10 {
		t.Fatalf("decode bar reset sampling progress: %+v", v)
	}
	p.line("[INFO] video.cpp:1944 - generate_video completed in 113.42s")
	if p.progress.Phase != "saving" || p.progress.Total != 0 {
		t.Fatalf("completed generation restarted sampling: %+v", p.progress)
	}
	p.line("[INFO] image.cpp:100 - generate_image 512x512")
	p.line("|=====>    | 1/20 - 2.00s/it")
	if p.progress.Unit != "steps" {
		t.Fatalf("next generation retained decode phase: %+v", p.progress)
	}
}

func TestPollSDJobBoundsAnHTTPServerThatNeverAnswers(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	m := newJobManager(t, &fakeSD{srv: srv})
	oldTimeout, oldBound := sdPollRequestTimeout, sdPollErrorBound
	sdPollRequestTimeout, sdPollErrorBound = 15*time.Millisecond, 25*time.Millisecond
	t.Cleanup(func() { sdPollRequestTimeout, sdPollErrorBound = oldTimeout, oldBound })
	start := time.Now()
	var unresponsive int
	_, err := m.pollSDJob(context.Background(), "model-1", (&fakeSD{srv: srv}).port(t), "hung", func(st string) error {
		if st == "unresponsive" {
			unresponsive++
		}
		return nil
	})
	if err == nil || unresponsive == 0 || time.Since(start) > time.Second {
		t.Fatalf("poll hung or hid failed heartbeats: err=%v ticks=%d elapsed=%s", err, unresponsive, time.Since(start))
	}
}

func TestRunStageWithProgressDeliversScopedJobProgress(t *testing.T) {
	f := newFakeSD(t, okPoll())
	m := newJobManager(t, f)
	var observed []Progress
	job, err := m.RunStageWithProgress(context.Background(), "model-1", GenerateParams{Prompt: "a boat"}, nil, func(p Progress) {
		observed = append(observed, p)
	})
	if err != nil || len(observed) == 0 {
		t.Fatalf("job=%+v err=%v progress=%v", job, err, observed)
	}
	for _, p := range observed {
		if p.JobID != job.ID {
			t.Fatalf("progress belonged to another job: %+v", p)
		}
	}
}

func TestPollSDJobBoundsAResponseBodyThatNeverFinishes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":`))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	m := newJobManager(t, &fakeSD{srv: srv})
	oldTimeout, oldBound := sdPollRequestTimeout, sdPollErrorBound
	sdPollRequestTimeout, sdPollErrorBound = 15*time.Millisecond, 25*time.Millisecond
	t.Cleanup(func() { sdPollRequestTimeout, sdPollErrorBound = oldTimeout, oldBound })
	start := time.Now()
	var unresponsive int
	_, err := m.pollSDJob(context.Background(), "model-1", (&fakeSD{srv: srv}).port(t), "hung-body", func(st string) error {
		if st == "unresponsive" {
			unresponsive++
		}
		return nil
	})
	if err == nil || unresponsive == 0 || time.Since(start) > time.Second {
		t.Fatalf("response body hung or hid heartbeat failures: %v ticks=%d elapsed=%s", err, unresponsive, time.Since(start))
	}
}

func TestSubmitSDJobBoundsAResponseBodyThatNeverFinishes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"id":`))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	m := newJobManager(t, &fakeSD{srv: srv})
	oldTimeout := sdSubmitRequestTimeout
	sdSubmitRequestTimeout = 20 * time.Millisecond
	t.Cleanup(func() { sdSubmitRequestTimeout = oldTimeout })
	start := time.Now()
	_, _, err := m.submitSDJob(context.Background(), (&fakeSD{srv: srv}).port(t), "/sdcpp/v1/vid_gen", map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "acknowledgement") || time.Since(start) > time.Second {
		t.Fatalf("submission hung: %v elapsed=%s", err, time.Since(start))
	}
}

func TestAssertionFailsRunningJobAndServerWithoutWaitingForProcessExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.log")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/sdcpp/v1/capabilities" {
			http.NotFound(w, r)
			return
		}
		if r.Method == http.MethodPost {
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"id":"asserted"}`))
			return
		}
		// The server can still answer polls after its GPU worker hits an
		// assertion, while gdb or a core dump delays actual process exit.
		_ = os.WriteFile(path, []byte("/src/ggml-cuda/binbcast.cu:293: GGML_ASSERT(nb10 % sizeof(src1_t) == 0) failed\n"), 0o644)
		_, _ = w.Write([]byte(`{"status":"generating"}`))
	}))
	t.Cleanup(srv.Close)
	m := newJobManager(t, &fakeSD{srv: srv})
	m.servers["model-1"].logFile = path
	j, err := m.RunStage(context.Background(), "model-1", GenerateParams{Prompt: "a boat"}, nil)
	if err == nil || j == nil || j.State != StateFailed || !strings.Contains(j.Error, "tensor layout") {
		t.Fatalf("asserted job kept running: job=%+v err=%v", j, err)
	}
	if sv := m.servers["model-1"]; sv.state != ServerFailed || sv.ready || sv.port != 0 {
		t.Fatalf("asserted server still available: %+v", sv)
	}
}

func TestConfirmedCrashFailsJobsAndIgnoresAReplacedServer(t *testing.T) {
	f := newFakeSD(t, okPoll())
	m := newJobManager(t, f)
	insertJob(t, m, "crashed-job", GenerateParams{Prompt: "a boat"})
	ctx, cancel := context.WithCancelCause(context.Background())
	m.jobs["crashed-job"] = &jobHandle{modelID: "model-1", cancel: cancel}
	old := m.servers["model-1"]
	reason := coreDumpFailure("/server.log")
	m.abortServer("model-1", old.port, reason)
	j, err := m.Get("crashed-job")
	if err != nil || j.State != StateFailed || j.Error != reason || context.Cause(ctx) != errServerCrashed {
		t.Fatalf("crash did not fail job with its cause: job=%+v err=%v cause=%v", j, err, context.Cause(ctx))
	}
	replacement := &server{modelID: "model-1", port: old.port + 1, ready: true, state: ServerReady}
	m.servers["model-1"] = replacement
	exited := make(chan struct{})
	close(exited)
	m.watchServerExit("model-1", old, exited)
	m.abortServer("model-1", replacement.port-1, reason)
	if !replacement.ready || replacement.state != ServerReady {
		t.Fatalf("old crash poisoned the replacement: %+v", replacement)
	}
}
