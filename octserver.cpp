#include <iostream>
#include <QtWidgets>
#include <QtNetwork>
#include <QDir>
#include "config.h"

#include "octserver.h"

// TODO: modify to send multiple volumes or to stream


OCTServer::OCTServer(QWidget *parent)
    : QDialog(parent)
    , statusLabel(new QLabel)
    , tcpServer(new QTcpServer)
{
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
    p_request_type = REQUEST_TILE;
    p_transfer_requested = false;

    initServer();
    auto quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);

    // Connections
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::close);
    connect(tcpServer, &QTcpServer::newConnection, this, &OCTServer::slot_startConnection);

    // Layout
    auto buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    QVBoxLayout *mainLayout = nullptr;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        auto outerVerticalLayout = new QVBoxLayout(this);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
        auto outerHorizontalLayout = new QHBoxLayout;
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        auto groupBox = new QGroupBox(QGuiApplication::applicationDisplayName());
        mainLayout = new QVBoxLayout(groupBox);
        outerHorizontalLayout->addWidget(groupBox);
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        outerVerticalLayout->addLayout(outerHorizontalLayout);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
    } else {
        mainLayout = new QVBoxLayout(this);
    }

    mainLayout->addWidget(statusLabel);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
}

OCTServer::~OCTServer()
{
    tcpServer->close();
}

void OCTServer::initServer()
{
    // Set the server address and port
    QString ipAddress = OCTServer::getHostAddress();
    const QHostAddress address = QHostAddress(ipAddress);
    quint16 ipPort = SERVER_SOCKET_PORT;

    if (!tcpServer->listen(address, ipPort)) {
        QMessageBox::critical(this, tr("OCT Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2\n\n"
                            "Waiting for connections ...")
                         .arg(ipAddress).arg(tcpServer->serverPort()));

}

void OCTServer::slot_endConnection()
{
    if (p_request_type == REQUEST_BSCAN_NOSAVE){ // This is done in another method
        return;
    }
    p_mutex.lock();
    if(p_transfer_requested){
        p_mutex.unlock();
        return;
    } else{
        p_transfer_requested=true;
        p_mutex.unlock();
    }
    std::cerr << "OCTServer::slot_endConnection ";
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    int n_bytes;
    out.setVersion(QDataStream::Qt_6_4);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);
    switch (p_request_type) {
    case REQUEST_TILE:
        n_bytes = 9 * sizeof(char) + 2 * sizeof(int); // QDataStream magic number + n_bytes + Message
        out << n_bytes;
        out << "OCT_done";
        break;
    case REQUEST_CONFIG:
        n_bytes = 16 * sizeof(char) + 2 * sizeof(int);
        out << n_bytes;
        out << "config:received";
        break;
    case REQUEST_UNKNOWN:
        n_bytes = 8 * sizeof(char) + 2 * sizeof(int);
        out << n_bytes;
        out << "unknown";
        break;
    }
    std::cerr << "(transmitting " << n_bytes << " bytes)... ";
    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
}

void OCTServer::slot_endConnectionAndSendImage(int nx, int ny, int nz, float* data)
{
    p_mutex.lock();
    if (p_transfer_requested)
    {
        p_mutex.unlock();
        return;
    } else {
        p_mutex.unlock();
        p_transfer_requested = true;
    }
    std::cerr << "OCTServer::slot_endConnectionAndSendImage ";

    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    out.setVersion(QDataStream::Qt_6_4);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);

    int n_bytes = nx * ny * nz * sizeof(float) + 4 * sizeof(int);
    std::cerr << "(transmitting " << n_bytes << " bytes)... ";

    // Add the data to the byte array
    out << n_bytes;
    out << nx;
    out << ny;
    out << nz;

    for (int i=0; i< nx * ny * nz; i++) {
       out << data[i];
    }

    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
}

QString OCTServer::getHostAddress()
{
    QString ipAddress;
    const QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();

    // use the first non-localhost IPv4 address
    for (const QHostAddress &entry : ipAddressesList) {
        if (entry != QHostAddress::LocalHost && entry.toIPv4Address()) {
            ipAddress = entry.toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return ipAddress;
}

// TODO: change the request names
void OCTServer::slot_parseRequest()
{
    p_mutex.lock();
    p_transfer_requested = false;
    p_mutex.unlock();
    QByteArray total_data, buffer;
    while(1) {
        buffer = clientConnection->read(1024);
        if (buffer.isEmpty()) {
            break;
        }
        total_data.append(buffer);
    }
    QString data = QString(total_data);

   if ( data.startsWith("request_autofocus_bscan"))
   {
       std::cerr << "Case 1: Autofocus B-Scan Request" << std::endl;
       p_request_type = REQUEST_BSCAN_NOSAVE;
       emit sig_set_request_type(QString("autofocus"));
       setup_and_request_scan();
   }
   else if ( data.startsWith("config")) {
       p_request_type = REQUEST_CONFIG;

       // Analyze the config
       QString config = data.split(":")[1];
       if (config == "nx"){
          int nx = data.split(":")[2].toInt();
          std::cerr << "OCTServer::slot_parseRequest|Setting nx=" << nx << std::endl;
          emit sig_config_nx(QString::number(nx));
       } else if (config == "ny") {
           int ny = data.split(":")[2].toInt();
           std::cerr << "OCTServer::slot_parseRequest|Setting ny=" << ny << std::endl;
           emit sig_config_ny(QString::number(ny));
       } else if (config == "fov_x") {
           int fov_x = data.split(":")[2].toFloat();
           std::cerr << "OCTServer::slot_parseRequest|Setting fov_x=" << fov_x << std::endl;
           emit sig_config_fov_x(QString::number(fov_x));
       } else if (config == "fov_y") {
           int fov_y = data.split(":")[2].toFloat();
           std::cerr << "OCTServer::slot_parseRequest|Setting fov_y=" << fov_y << std::endl;
           emit sig_config_fov_y(QString::number(fov_y));
       } else if (config == "x_mm") {
           float x_mm = data.split(":")[2].toFloat();
           float y_mm = data.split(":")[4].toFloat();
           float z_mm = data.split(":")[6].toFloat();
           std::cerr << "OCTServer::slot_parseRequest|Setting x_mm=" << x_mm << ", y_mm=" << y_mm << ", z_mm=" << z_mm << std::endl;
           emit sig_config_pos(x_mm, y_mm, z_mm);
       }
       else {
           std::cerr << "OCTServer::slot_parseRequest|Unknown command:" << data.toStdString() << std::endl;
           p_request_type = REQUEST_UNKNOWN;
       }
       slot_endConnection();
   }
   else
   {
       std::cerr << "Case 2: Tile Request" << std::endl;
       p_request_type = REQUEST_TILE;
       emit sig_set_request_type(QString("tile"));
       std::cerr << "octserver::Reading tile positions: (x,y,z)=";
       // Read the tile position to acquire
       p_tile_x = data.split(" ")[0].toInt();
       p_tile_y = data.split(" ")[1].toInt();
       p_tile_z = data.split(" ")[2].toInt();

       std::cerr << "(" << p_tile_x <<"," << p_tile_y << "," << p_tile_z << ")" << std::endl;
        setup_and_request_scan(p_tile_x, p_tile_y, p_tile_z);
   }
}

void OCTServer::slot_startConnection()
{
    std::cerr << "OCTServer::slot_startConnection" << std::endl;
    clientConnection = tcpServer->nextPendingConnection();
    connect(clientConnection, &QIODevice::readyRead, this, &OCTServer::slot_parseRequest);
}

void OCTServer::setup_and_request_scan(int x, int y, int z)
{
    // Create the tile filename
    char buffer [20];
    snprintf(buffer, 20, "tile_x%02d_y%02d_z%02d", x, y, z);
    QString fileName = QString(buffer);
    emit sig_change_filename(fileName);

    // Launch a single acquisition
    emit sig_start_scan();
}

void OCTServer::setup_and_request_scan()
{
    // Create the tile filename
    QString fileName = QString("dummy");
    emit sig_change_filename(fileName);

    // Launch a single acquisition
    emit sig_start_scan();
}

