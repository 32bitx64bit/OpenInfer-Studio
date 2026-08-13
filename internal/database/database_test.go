package database

import (
	"testing"

	"github.com/openinfer/openinfer-studio/migrations"
)

func TestMigrationsApply(t *testing.T) {
	db, err := Open(t.TempDir(), migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Spot-check key tables exist.
	for _, table := range []string{"settings", "models", "runtimes", "downloads",
		"download_files", "instances", "conversations", "conversation_messages",
		"model_presets", "server_profiles", "diagnostic_events",
		"quant_jobs", "imatrices"} {
		var name string
		err := db.QueryRow(`SELECT name FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&name)
		if err != nil {
			t.Errorf("table %s missing: %v", table, err)
		}
	}
}

func TestMigrationsIdempotent(t *testing.T) {
	dir := t.TempDir()
	db1, err := Open(dir, migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	db1.Close()
	db2, err := Open(dir, migrations.FS) // second open must not re-apply
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	var n int
	db2.QueryRow(`SELECT COUNT(1) FROM schema_migrations`).Scan(&n)
	if n != 2 {
		t.Errorf("migrations recorded = %d, want 2", n)
	}
}

func TestForeignKeysEnforced(t *testing.T) {
	db, err := Open(t.TempDir(), migrations.FS)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	// Insert into a child table referencing a missing model must fail.
	_, err = db.Exec(`INSERT INTO model_files(id,model_id,path) VALUES ('x','missing','/f.gguf')`)
	if err == nil {
		t.Error("foreign key violation not enforced")
	}
}
