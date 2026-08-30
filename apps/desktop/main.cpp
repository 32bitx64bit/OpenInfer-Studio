// OpenInfer Studio — minimal Qt bootstrap.
// Launches openinfer-core, passes token/port into QML, kills the backend on exit.
// Application logic lives in the Go backend; keep this file free of it.

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QProcess>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QRandomGenerator>
#include <QSaveFile>
#include <QTcpServer>
#include <QTimer>

#ifndef OPENINFER_VERSION
#define OPENINFER_VERSION "0.0.0-dev"
#endif

namespace {

QString generateToken()
{
    QByteArray raw(16, 0);
    QRandomGenerator::system()->fillRange(reinterpret_cast<quint32*>(raw.data()), 4);
    return raw.toHex();
}

quint16 pickFreePort()
{
    QTcpServer probe;
    if (!probe.listen(QHostAddress::LocalHost, 0))
        return 0;
    quint16 port = probe.serverPort();
    probe.close();
    return port;
}

QString locateBackend(const QString &appDir)
{
#ifdef Q_OS_WIN
    const QString name = QStringLiteral("openinfer-core.exe");
#else
    const QString name = QStringLiteral("openinfer-core");
#endif
    const QString env = qEnvironmentVariable("OPENINFER_CORE");
    if (!env.isEmpty() && QFileInfo::exists(env))
        return env;
    const QStringList candidates = {
        appDir + QLatin1Char('/') + name,
        appDir + QStringLiteral("/../core/") + name,
        appDir + QStringLiteral("/core/") + name,
    };
    for (const QString &c : candidates)
        if (QFileInfo::exists(c))
            return QDir(c).canonicalPath();
    return {};
}

int fatal(const QString &title, const QString &message)
{
    QMessageBox box(QMessageBox::Critical, title, message, QMessageBox::Ok);
    box.exec();
    return 1;
}

#ifdef Q_OS_LINUX
// Quote a path for a .desktop Exec= key (desktop-entry spec quoting + %% ).
QString quoteDesktopExec(const QString &path)
{
    QString escaped = path;
    escaped.replace(QLatin1Char('\\'), QStringLiteral("\\\\"));
    escaped.replace(QLatin1Char('"'), QStringLiteral("\\\""));
    escaped.replace(QLatin1Char('`'), QStringLiteral("\\`"));
    escaped.replace(QLatin1Char('$'), QStringLiteral("\\$"));
    escaped.replace(QLatin1Char('%'), QStringLiteral("%%"));
    return QLatin1Char('"') + escaped + QLatin1Char('"');
}

// AppImages mount at a new /tmp/.mount_* path every launch. Plasma pins the
// in-image .desktop at that ephemeral path, then fails on the next click.
// Install a stable user launcher that Exec=s the AppImage itself, matching
// setDesktopFileName("openinfer-studio"), before the window is created.
void installAppImageDesktopEntry()
{
    const QString appImage = qEnvironmentVariable("APPIMAGE");
    if (appImage.isEmpty() || appImage.contains(QLatin1Char('\n')))
        return;
    if (!QFileInfo::exists(appImage))
        return;

    const QString dataHome = qEnvironmentVariable("XDG_DATA_HOME");
    const QString share = dataHome.isEmpty()
        ? QDir::homePath() + QStringLiteral("/.local/share")
        : dataHome;
    const QString appsDir = share + QStringLiteral("/applications");
    if (!QDir().mkpath(appsDir))
        return;

    QString iconValue = QStringLiteral("openinfer-studio");
    const QString appDir = qEnvironmentVariable("APPDIR");
    if (!appDir.isEmpty()) {
        const QStringList iconCandidates = {
            appDir + QStringLiteral("/usr/share/icons/hicolor/256x256/apps/openinfer-studio.png"),
            appDir + QStringLiteral("/usr/share/icons/hicolor/128x128/apps/openinfer-studio.png"),
            appDir + QStringLiteral("/openinfer-studio.png"),
            appDir + QStringLiteral("/usr/share/icons/hicolor/scalable/apps/openinfer-studio.svg"),
            appDir + QStringLiteral("/usr/share/icons/hicolor/256x256/apps/openinfer-studio.svg"),
            appDir + QStringLiteral("/openinfer-studio.svg"),
        };
        for (const QString &src : iconCandidates) {
            if (!QFileInfo::exists(src))
                continue;
            const bool svg = src.endsWith(QLatin1String(".svg"));
            const QString rel = svg
                ? QStringLiteral("/icons/hicolor/scalable/apps")
                : QStringLiteral("/icons/hicolor/256x256/apps");
            const QString iconDir = share + rel;
            if (!QDir().mkpath(iconDir))
                break;
            const QString dst = iconDir + (svg
                ? QStringLiteral("/openinfer-studio.svg")
                : QStringLiteral("/openinfer-studio.png"));
            QFile::remove(dst);
            if (QFile::copy(src, dst))
                iconValue = dst;
            break;
        }
    }

    QString body;
    body += QStringLiteral("[Desktop Entry]\n");
    body += QStringLiteral("Name=OpenInfer Studio\n");
    body += QStringLiteral("Comment=Run GGUF models locally with llama.cpp\n");
    body += QStringLiteral("Exec=") + quoteDesktopExec(appImage) + QLatin1Char('\n');
    body += QStringLiteral("TryExec=") + appImage + QLatin1Char('\n');
    body += QStringLiteral("Icon=") + iconValue + QLatin1Char('\n');
    body += QStringLiteral("Type=Application\n");
    body += QStringLiteral("Categories=Development;Utility;\n");
    body += QStringLiteral("Terminal=false\n");
    body += QStringLiteral("StartupWMClass=openinfer-studio\n");
    body += QStringLiteral("X-AppImage-Version=") + QStringLiteral(OPENINFER_VERSION) + QLatin1Char('\n');
    body += QStringLiteral("X-OpenInfer-AppImage=") + appImage + QLatin1Char('\n');

    const QString desktopPath = appsDir + QStringLiteral("/openinfer-studio.desktop");
    QFile existing(desktopPath);
    if (existing.open(QIODevice::ReadOnly) && existing.readAll() == body.toUtf8())
        return;

    QSaveFile out(desktopPath);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
    out.write(body.toUtf8());
    out.commit();
}
#endif

} // namespace

int main(int argc, char *argv[])
{
#ifdef Q_OS_LINUX
    installAppImageDesktopEntry();
#endif

    // High-DPI and application identity.
    QGuiApplication::setApplicationName(QStringLiteral("openinfer-studio"));
    QGuiApplication::setApplicationDisplayName(QStringLiteral("OpenInfer Studio"));
    QGuiApplication::setOrganizationName(QStringLiteral("OpenInfer"));
    QGuiApplication::setApplicationVersion(QStringLiteral(OPENINFER_VERSION));
    // Ties the window to openinfer-studio.desktop (X11 WM_CLASS / Wayland
    // app_id). Required for taskbar pinning to match the running window on
    // KDE/GNOME; must stay in sync with StartupWMClass in the .desktop file.
    QGuiApplication::setDesktopFileName(QStringLiteral("openinfer-studio"));

    // Native styles (macOS/Windows) and Plasma's org.kde.breeze reject or
    // mishandle control customization used throughout the QML UI. Force Fusion
    // even when QT_QUICK_CONTROLS_STYLE is set in the environment.
    qputenv("QT_QUICK_CONTROLS_STYLE", "Fusion");
    QQuickStyle::setStyle(QStringLiteral("Fusion"));

    QGuiApplication app(argc, argv);

    const QString token = generateToken();
    const quint16 port = pickFreePort();
    if (port == 0)
        return fatal(QStringLiteral("OpenInfer Studio"),
                     QStringLiteral("Could not allocate a local port for the backend."));

    const QString backendPath = locateBackend(QCoreApplication::applicationDirPath());
    if (backendPath.isEmpty())
        return fatal(QStringLiteral("OpenInfer Studio"),
                     QStringLiteral("The backend executable (openinfer-core) was not found.\n\n"
                                    "Reinstall the application or set OPENINFER_CORE for development."));

    QProcess backend;
    backend.setProgram(backendPath);
    backend.setArguments({
        QStringLiteral("--token"), token,
        QStringLiteral("--port"), QString::number(port),
        QStringLiteral("--ppid"), QString::number(QCoreApplication::applicationPid()),
    });
    // Merge stderr into stdout so diagnostics arrive in one stream.
    backend.setProcessChannelMode(QProcess::MergedChannels);
    backend.start();

    // Backend prints {"ready":true,...}; keep extra output for fatal dialogs.
    QByteArray backendOutput;
    QObject::connect(&backend, &QProcess::readyRead, &app, [&] {
        backendOutput += backend.readAll();
    });

    if (!backend.waitForStarted(10000))
        return fatal(QStringLiteral("OpenInfer Studio"),
                     QStringLiteral("Failed to start the backend:\n%1").arg(backend.errorString()));

    // Up to 30s: first-run migrations run before the readiness line.
    bool ready = false;
    {
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        QObject::connect(&backend, &QProcess::readyRead, &loop, &QEventLoop::quit);
        QObject::connect(&backend, &QProcess::finished, &loop, &QEventLoop::quit);
        timeout.start(30000);
        while (!ready && timeout.isActive() && backend.state() == QProcess::Running) {
            loop.exec();
            if (backendOutput.contains("\"ready\":true")) {
                ready = true;
                break;
            }
            if (backendOutput.contains("\"ready\":false"))
                break;
        }
    }
    if (!ready) {
        QString detail = QString::fromUtf8(backendOutput).trimmed();
        if (detail.size() > 2000)
            detail = detail.right(2000);
        backend.kill();
        backend.waitForFinished(3000);
        return fatal(QStringLiteral("OpenInfer Studio"),
                     QStringLiteral("The backend did not become ready.\n\nBackend output:\n%1")
                         .arg(detail.isEmpty() ? QStringLiteral("(no output)") : detail));
    }

    // Kill backend with the UI; parent-death watchdog is the backup.
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &app, [&] {
        backend.terminate();
        if (!backend.waitForFinished(4000))
            backend.kill();
    });

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty(QStringLiteral("apiBase"),
                                             QStringLiteral("http://127.0.0.1:%1").arg(port));
    engine.rootContext()->setContextProperty(QStringLiteral("wsBase"),
                                             QStringLiteral("ws://127.0.0.1:%1").arg(port));
    engine.rootContext()->setContextProperty(QStringLiteral("apiToken"), token);
    engine.rootContext()->setContextProperty(QStringLiteral("appVersion"),
                                             QStringLiteral(OPENINFER_VERSION));

    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
                     [] { QCoreApplication::exit(2); }, Qt::QueuedConnection);
    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    return app.exec();
}
